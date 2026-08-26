#include <catch2/catch_test_macros.hpp>

#include <vkexec/buffer.hpp>
#include <vkexec/bulk.hpp>
#include <vkexec/context.hpp>
#include <vkexec/domain.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_edsl/push_constant.hpp>
#include <vkexec_edsl/types.hpp>

#include <boost/describe/class.hpp>

#include <stdexec/execution.hpp>

#include <cmath>
#include <concepts>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

namespace ex = stdexec;
namespace edsl = vkexec::edsl;

namespace {

struct env_params
{
  float n;
};
// cppcheck-suppress unknownMacro
BOOST_DESCRIBE_STRUCT(env_params, (), (n))

auto skip_if_no_vulkan(vkexec::error const &err) -> void
{ SKIP(std::string("Vulkan unavailable: ") + std::string(err.message())); }

}// namespace

TEST_CASE("schedule_sender advertises completion scheduler", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto const sender = sched.schedule();

  // NOLINTNEXTLINE(misc-include-cleaner)
  auto const completion = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
  REQUIRE(completion == sched);
  REQUIRE(completion.get_context() == nullptr);
}

TEST_CASE("schedule completes on the context host agent", "[vkexec][scheduler][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto const caller = std::this_thread::get_id();
  auto const agent = ctx.host_agent_thread_id();
  REQUIRE(agent != caller);

  std::thread::id completed_on{};
  auto waited = vkexec::sync_wait(
    ex::schedule(ctx.get_scheduler()) | ex::then([&]() -> void { completed_on = std::this_thread::get_id(); }));
  REQUIRE(waited.has_value());
  REQUIRE(waited->has_value());

  REQUIRE(completed_on == agent);
  REQUIRE(completed_on != caller);
}

TEST_CASE("starts_on runs the child on the context host agent", "[vkexec][scheduler][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto const caller = std::this_thread::get_id();
  auto const agent = ctx.host_agent_thread_id();

  std::thread::id ran_on{};
  auto waited = vkexec::sync_wait(
    ex::starts_on(ctx.get_scheduler(), ex::just() | ex::then([&]() -> void { ran_on = std::this_thread::get_id(); })));
  REQUIRE(waited.has_value());
  REQUIRE(waited->has_value());

  REQUIRE(ran_on == agent);
  REQUIRE(ran_on != caller);
}

TEST_CASE("schedule_sender advertises completion domain", "[vkexec][scheduler][domain]")
{
  vkexec::scheduler_env const env{};
  STATIC_REQUIRE(std::same_as<decltype(env.query(ex::get_completion_domain_t<ex::set_value_t>{})), vkexec::domain>);
  STATIC_REQUIRE(std::same_as<decltype(env.query(ex::get_domain_t{})), vkexec::domain>);

  vkexec::scheduler const sched{ nullptr };
  STATIC_REQUIRE(std::same_as<decltype(sched.query(ex::get_completion_domain_t<ex::set_value_t>{})), vkexec::domain>);
  STATIC_REQUIRE(std::same_as<decltype(sched.query(ex::get_domain_t{})), vkexec::domain>);
}

TEST_CASE("starts_on then bulk lowers via vkexec domain", "[vkexec][scheduler][domain][gpu]")
{
  constexpr std::uint32_t k_count = 8;
  constexpr float k_initial = 1.0F;
  constexpr float k_factor = 2.0F;
  constexpr float k_epsilon = 1.0E-4F;

  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto values_result = vkexec::buffer<float>::create_sync(ctx, k_count, k_initial);
  REQUIRE(values_result.has_value());
  auto &values = *values_result;

  auto waited = vkexec::sync_wait(
    ex::starts_on(ctx.get_scheduler(), ex::just())
    | vkexec::bulk(
      k_count, env_params{ .n = k_factor }, [&](edsl::Int idx, edsl::push_constant<env_params> push) -> void {
        values[idx] = values[idx] * push.get<&env_params::n>();
      }));
  REQUIRE(waited.has_value());
  REQUIRE(waited->has_value());

  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  REQUIRE(std::fabs(values.data()[0] - (k_initial * k_factor)) < k_epsilon);
}

TEST_CASE("bulk_sender advertises completion scheduler", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto const sender =
    ex::schedule(sched)
    | vkexec::bulk(
      1U, env_params{ .n = 1.0F }, [](edsl::Int /*idx*/, edsl::push_constant<env_params> /*push*/) -> void {});

  // NOLINTNEXTLINE(misc-include-cleaner)
  auto const completion = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
  REQUIRE(completion == sched);
}

TEST_CASE("bulk_async_sender advertises completion scheduler", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto const sender =
    ex::schedule(sched)
    | vkexec::bulk(
      1U, env_params{ .n = 1.0F }, [](edsl::Int /*idx*/, edsl::push_constant<env_params> /*push*/) -> void {})
    | vkexec::submit;

  // NOLINTNEXTLINE(misc-include-cleaner)
  auto const completion = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
  REQUIRE(completion == sched);
}

TEST_CASE("pass_graph_sender advertises completion scheduler", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto const sender =
    ex::schedule(sched)
    | vkexec::compute_pass(
      1U, env_params{ .n = 1.0F }, [](edsl::Int /*idx*/, edsl::push_constant<env_params> /*push*/) -> void {});

  // NOLINTNEXTLINE(misc-include-cleaner)
  auto const completion = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
  REQUIRE(completion == sched);
}

TEST_CASE("pass_graph_async_sender advertises completion scheduler", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto const sender =
    ex::schedule(sched)
    | vkexec::compute_pass(
      1U, env_params{ .n = 1.0F }, [](edsl::Int /*idx*/, edsl::push_constant<env_params> /*push*/) -> void {})
    | vkexec::submit;

  // NOLINTNEXTLINE(misc-include-cleaner)
  auto const completion = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
  REQUIRE(completion == sched);
}
