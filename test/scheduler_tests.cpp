#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <stdexec/__detail/__execution_fwd.hpp>
#include <vkexec/context.hpp>
#include <vkexec/domain.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit.hpp>

#include <stdexec/execution.hpp>

#include <concepts>
#include <memory>
#include <thread>

namespace ex = stdexec;

TEST_CASE("schedule_sender advertises completion scheduler", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto const sender = sched.schedule();

  auto const completion = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
  REQUIRE(completion == sched);
  REQUIRE(completion.get_context() == nullptr);
}

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

TEST_CASE("pass_graph_sender advertises completion scheduler", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto const sender = ex::schedule(sched) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 });

  auto const completion = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
  REQUIRE(completion == sched);
}

TEST_CASE("pass_graph_async_sender advertises completion scheduler", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto const sender =
    ex::schedule(sched) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 }) | vkexec::submit;

  auto const completion = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
  REQUIRE(completion == sched);
}
