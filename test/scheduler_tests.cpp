#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/context.hpp>
#include <vkexec/domain.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/result.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/schema_pass.hpp>
#include <vkexec/submit_scope.hpp>
#include <vkexec/tensor_pass.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <atomic>
#include <chrono>
#include <concepts>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <thread>
#include <utility>

namespace ex = stdexec;

namespace {

struct empty_receiver_env
{
};

auto signal_promise(std::promise<void> &promise) noexcept -> void
{
  try {
    promise.set_value();
  } catch (std::future_error const &err) {
    (void)err;// already satisfied
  }
}

struct completion_probe_receiver
{
  using receiver_concept = ex::receiver_t;

  std::promise<void> *completed{ nullptr };

  auto signal_done() const noexcept -> void { signal_promise(*completed); }

  auto set_value() const && noexcept -> void { signal_done(); }

  auto set_error(vkexec::error const & /*err*/) const && noexcept -> void { signal_done(); }

  auto set_error(std::exception_ptr const & /*exception*/) const && noexcept -> void { signal_done(); }

  auto set_stopped() const && noexcept -> void { signal_done(); }

  // cppcheck-suppress functionStatic
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  [[nodiscard]] auto get_env() const noexcept -> empty_receiver_env { return {}; }
};

[[nodiscard]] auto make_empty_pass_step(std::function<void()> after_gpu) -> vkexec::pass_step
{
  return vkexec::pass_step{
    .record = [](vkexec::context & /*host*/,
                VkCommandBuffer /*cmd*/,
                vkexec::detail::pass_cleanup & /*cleanup*/) -> vkexec::status { return {}; },
    .after_gpu = std::move(after_gpu),
  };
}

struct release_promise
{
  std::promise<void> *gate{ nullptr };

  explicit release_promise(std::promise<void> &promise) noexcept : gate(&promise) {}

  release_promise(release_promise const &) = delete;
  auto operator=(release_promise const &) -> release_promise & = delete;
  release_promise(release_promise &&) = delete;
  auto operator=(release_promise &&) -> release_promise & = delete;

  auto release() noexcept -> void
  {
    if (gate == nullptr) { return; }
    signal_promise(*gate);
    gate = nullptr;
  }

  ~release_promise() { release(); }
};

}// namespace

template<class Sender, class CompletionTag>
concept advertises_completion_scheduler =
  requires(Sender const &sndr) { ex::get_completion_scheduler<CompletionTag>(ex::get_env(sndr)); };

using pass_adaptor_t = vkexec::pass_adaptor_sender<vkexec::schedule_sender, vkexec::prebuilt_compute_pass_closure>;
using schema_pass_adaptor_t = vkexec::schema_pass_sender<vkexec::schedule_sender>;
using tensor_pass_adaptor_t = vkexec::tensor_pass_sender<vkexec::schedule_sender, vkexec::tensor_pass_closure<float>>;

static_assert(advertises_completion_scheduler<vkexec::schedule_sender, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::schedule_sender, ex::set_error_t>);

static_assert(advertises_completion_scheduler<vkexec::pass_graph_sender, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::pass_graph_sender, ex::set_error_t>);

static_assert(!advertises_completion_scheduler<vkexec::detail::enter_submit_scope_sender, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::detail::submit_and_wait_sender, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::detail::submit_fence_sender, ex::set_value_t>);

static_assert(advertises_completion_scheduler<pass_adaptor_t, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<pass_adaptor_t, ex::set_error_t>);
static_assert(advertises_completion_scheduler<schema_pass_adaptor_t, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<schema_pass_adaptor_t, ex::set_error_t>);
static_assert(advertises_completion_scheduler<tensor_pass_adaptor_t, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<tensor_pass_adaptor_t, ex::set_error_t>);

static_assert(vkexec::vkexec_predecessor<vkexec::schedule_sender>);
static_assert(vkexec::vkexec_predecessor<vkexec::pass_graph_sender>);
static_assert(vkexec::vkexec_predecessor<pass_adaptor_t>);
static_assert(vkexec::vkexec_predecessor<schema_pass_adaptor_t>);
static_assert(vkexec::vkexec_predecessor<tensor_pass_adaptor_t>);

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

  STATIC_REQUIRE(std::same_as<decltype(env.query(ex::get_completion_domain_t<ex::set_value_t>{})), vkexec::domain>);
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

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("pass_graph_sender start returns before GPU completion", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();

  std::promise<void> after_gpu_entered;
  auto after_gpu_ready = after_gpu_entered.get_future();
  std::promise<void> allow_completion;
  auto allow_completion_future = allow_completion.get_future();
  release_promise release_gate{ allow_completion };
  std::promise<void> receiver_completed;
  auto receiver_done = receiver_completed.get_future();

  vkexec::pass_graph_sender graph{
    .ctx = ctx.get(),
    .steps =
      {
        make_empty_pass_step([&]() noexcept -> void {
          signal_promise(after_gpu_entered);
          allow_completion_future.wait();
        }),
      },
  };

  auto operation = ex::connect(std::move(graph), completion_probe_receiver{ .completed = &receiver_completed });

  std::promise<void> start_returned;
  auto start_done = start_returned.get_future();

  std::jthread starter{ [&]() -> void {
    ex::start(operation);
    signal_promise(start_returned);
  } };

  auto const after_gpu_status = after_gpu_ready.wait_for(std::chrono::seconds(5));
  auto const start_status = start_done.wait_for(std::chrono::seconds(2));
  auto const receiver_status_before_release = receiver_done.wait_for(std::chrono::seconds(0));

  release_gate.release();
  auto const receiver_status_after_release = receiver_done.wait_for(std::chrono::seconds(5));
  starter.join();

  REQUIRE(after_gpu_status == std::future_status::ready);
  REQUIRE(start_status == std::future_status::ready);
  REQUIRE(receiver_status_before_release == std::future_status::timeout);
  REQUIRE(receiver_status_after_release == std::future_status::ready);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("sync_wait still waits for pass_graph_sender completion", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();

  std::promise<void> after_gpu_entered;
  auto after_gpu_ready = after_gpu_entered.get_future();
  std::promise<void> allow_completion;
  auto allow_completion_future = allow_completion.get_future();
  release_promise release_gate{ allow_completion };
  std::atomic_bool sync_wait_returned{ false };
  std::atomic_bool completed_successfully{ false };

  vkexec::pass_graph_sender graph{
    .ctx = ctx.get(),
    .steps =
      {
        make_empty_pass_step([&]() noexcept -> void {
          signal_promise(after_gpu_entered);
          allow_completion_future.wait();
        }),
      },
  };

  std::jthread waiter{ [&]() -> void {
    auto const waited = vkexec::test::sync_wait_sender(std::move(graph));
    completed_successfully.store(vkexec::test::sync_wait_completed(waited), std::memory_order_release);
    sync_wait_returned.store(true, std::memory_order_release);
  } };

  auto const after_gpu_status = after_gpu_ready.wait_for(std::chrono::seconds(5));
  auto const returned_while_blocked = sync_wait_returned.load(std::memory_order_acquire);

  release_gate.release();
  waiter.join();

  auto const returned_after_release = sync_wait_returned.load(std::memory_order_acquire);
  auto const success = completed_successfully.load(std::memory_order_acquire);

  REQUIRE(after_gpu_status == std::future_status::ready);
  REQUIRE_FALSE(returned_while_blocked);
  REQUIRE(returned_after_release);
  REQUIRE(success);
}

TEST_CASE("pass_graph_sender completes on the context host scheduler", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto const agent = ctx->host_agent_thread_id();

  std::thread::id completed_on{};
  vkexec::pass_graph_sender graph{
    .ctx = ctx.get(),
    .steps = { make_empty_pass_step({}) },
  };

  auto waited = vkexec::test::sync_wait_sender(
    std::move(graph) | ex::then([&]() noexcept -> void { completed_on = std::this_thread::get_id(); }));
  REQUIRE(vkexec::test::sync_wait_completed(waited));
  REQUIRE(completed_on == agent);
}

TEST_CASE("pass composition uses one graph sender type", "[vkexec][pass]")
{
  vkexec::scheduler sched{ nullptr };

  auto graph = ex::schedule(sched) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 });
  STATIC_REQUIRE(std::same_as<decltype(graph), vkexec::pass_graph_sender>);

  auto graph2 = std::move(graph) | vkexec::barrier::compute_to_compute();
  STATIC_REQUIRE(std::same_as<decltype(graph2), vkexec::pass_graph_sender>);

  auto graph3 = std::move(graph2) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 });
  STATIC_REQUIRE(std::same_as<decltype(graph3), vkexec::pass_graph_sender>);
}
