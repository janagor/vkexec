#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/domain.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/schema_pass.hpp>
#include <vkexec/submit_scope.hpp>
#include <vkexec/tensor_pass.hpp>

#include <stdexec/execution.hpp>

#include <concepts>
#include <memory>
#include <thread>
#include <utility>

namespace ex = stdexec;

template<class Sender, class CompletionTag>
concept advertises_completion_scheduler = requires(Sender const &sndr) {
  ex::get_completion_scheduler<CompletionTag>(ex::get_env(sndr));
};

using pass_adaptor_t =
  vkexec::pass_adaptor_sender<vkexec::schedule_sender, vkexec::prebuilt_compute_pass_closure>;
using pass_async_adaptor_t =
  vkexec::pass_async_adaptor_sender<vkexec::schedule_sender, vkexec::prebuilt_compute_pass_closure>;
using schema_pass_adaptor_t = vkexec::schema_pass_sender<vkexec::schedule_sender>;
using tensor_pass_adaptor_t =
  vkexec::tensor_pass_sender<vkexec::schedule_sender, vkexec::tensor_pass_closure<float>>;

static_assert(advertises_completion_scheduler<vkexec::schedule_sender, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::schedule_sender, ex::set_error_t>);

static_assert(!advertises_completion_scheduler<vkexec::pass_graph_sender, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::pass_graph_async_sender, ex::set_value_t>);

static_assert(!advertises_completion_scheduler<vkexec::detail::enter_submit_scope_sender, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::detail::submit_and_wait_sender, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::detail::submit_fence_sender, ex::set_value_t>);

static_assert(advertises_completion_scheduler<pass_adaptor_t, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<pass_adaptor_t, ex::set_error_t>);
static_assert(advertises_completion_scheduler<schema_pass_adaptor_t, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<schema_pass_adaptor_t, ex::set_error_t>);
static_assert(advertises_completion_scheduler<tensor_pass_adaptor_t, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<tensor_pass_adaptor_t, ex::set_error_t>);
static_assert(!advertises_completion_scheduler<pass_async_adaptor_t, ex::set_value_t>);

static_assert(vkexec::vkexec_predecessor<vkexec::schedule_sender>);
static_assert(!vkexec::vkexec_predecessor<vkexec::pass_graph_sender>);
static_assert(!vkexec::vkexec_predecessor<vkexec::pass_graph_async_sender>);
static_assert(vkexec::vkexec_predecessor<pass_adaptor_t>);
static_assert(vkexec::vkexec_predecessor<schema_pass_adaptor_t>);
static_assert(vkexec::vkexec_predecessor<tensor_pass_adaptor_t>);
static_assert(!vkexec::vkexec_predecessor<pass_async_adaptor_t>);

TEST_CASE("schedule_sender advertises completion scheduler", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto const sender = sched.schedule();

  auto const completion = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
  REQUIRE(completion == sched);
  REQUIRE(completion.get_context() == nullptr);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("schedule completes on the context host agent", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();

  auto const caller = std::this_thread::get_id();
  auto const agent = ctx->host_agent_thread_id();
  REQUIRE(agent != caller);

  std::thread::id completed_on{};
  auto waited = vkexec::test::sync_wait_sender(
    ex::schedule(ctx->get_scheduler()) | ex::then([&]() -> void { completed_on = std::this_thread::get_id(); }));
  REQUIRE(vkexec::test::sync_wait_completed(waited));

  REQUIRE(completed_on == agent);
  REQUIRE(completed_on != caller);
}

TEST_CASE("starts_on runs the child on the context host agent", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();

  auto const caller = std::this_thread::get_id();
  auto const agent = ctx->host_agent_thread_id();

  std::thread::id ran_on{};
  auto waited = vkexec::test::sync_wait_sender(
    ex::starts_on(ctx->get_scheduler(), ex::just() | ex::then([&]() -> void { ran_on = std::this_thread::get_id(); })));
  REQUIRE(vkexec::test::sync_wait_completed(waited));

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

TEST_CASE("domain-only sender environment advertises vkexec domain", "[vkexec][scheduler][domain]")
{
  vkexec::detail::domain_env const env{};

  STATIC_REQUIRE(
    std::same_as<decltype(env.query(ex::get_completion_domain_t<ex::set_value_t>{})), vkexec::domain>);
  STATIC_REQUIRE(std::same_as<decltype(env.query(ex::get_domain_t{})), vkexec::domain>);
}

TEST_CASE("pass_adaptor_sender preserves predecessor value completion scheduler", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto const closure = vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 });
  pass_adaptor_t const sndr{
    .pred = sched.schedule(),
    .closure = closure,
  };

  auto const completion = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr));
  REQUIRE(completion == sched);
}

TEST_CASE("schema_pass_sender preserves predecessor value completion scheduler", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  schema_pass_adaptor_t const sndr{
    .pred = sched.schedule(),
    .closure = {},
  };

  auto const completion = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr));
  REQUIRE(completion == sched);
}

TEST_CASE("tensor_pass_sender preserves predecessor value completion scheduler", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  tensor_pass_adaptor_t const sndr{
    .pred = sched.schedule(),
    .closure = {},
  };

  auto const completion = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr));
  REQUIRE(completion == sched);
}

TEST_CASE("pass_async_adaptor_sender does not advertise completion scheduler", "[vkexec][scheduler]")
{
  STATIC_REQUIRE(!advertises_completion_scheduler<pass_async_adaptor_t, ex::set_value_t>);
  STATIC_REQUIRE(
    std::same_as<decltype(std::declval<pass_async_adaptor_t const &>().get_env()), vkexec::detail::domain_env>);
}
