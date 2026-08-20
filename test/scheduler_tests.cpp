#include <catch2/catch_test_macros.hpp>

#include <vkexec/bulk.hpp>
#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit.hpp>
#include <vkexec_edsl/push_constant.hpp>
#include <vkexec_edsl/types.hpp>

#include <boost/describe/class.hpp>

#include <stdexec/execution.hpp>

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
  vkexec::context ctx;
  auto const caller = std::this_thread::get_id();
  auto const agent = ctx.host_agent_thread_id();
  REQUIRE(agent != caller);

  std::thread::id completed_on{};
  ex::sync_wait(
    ex::schedule(ctx.get_scheduler()) | ex::then([&]() -> void { completed_on = std::this_thread::get_id(); }));

  REQUIRE(completed_on == agent);
  REQUIRE(completed_on != caller);
}

TEST_CASE("starts_on runs the child on the context host agent", "[vkexec][scheduler][gpu]")
{
  vkexec::context ctx;
  auto const caller = std::this_thread::get_id();
  auto const agent = ctx.host_agent_thread_id();

  std::thread::id ran_on{};
  ex::sync_wait(ex::starts_on(
    ctx.get_scheduler(), ex::just() | ex::then([&]() -> void { ran_on = std::this_thread::get_id(); })));

  REQUIRE(ran_on == agent);
  REQUIRE(ran_on != caller);
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
