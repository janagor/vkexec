#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/bind_resources.hpp>
#include <vkexec/context.hpp>
#include <vkexec/domain.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
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
#include <future>
#include <memory>
#include <thread>
#include <tuple>
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

struct move_only_pass_step
{
  std::unique_ptr<int> state;

  explicit move_only_pass_step(int initial_value) : state(std::make_unique<int>(initial_value)) {}
  ~move_only_pass_step() = default;
  move_only_pass_step(move_only_pass_step &&) noexcept = default;
  auto operator=(move_only_pass_step &&) noexcept -> move_only_pass_step & = default;
  move_only_pass_step(move_only_pass_step const &) = delete;
  auto operator=(move_only_pass_step const &) -> move_only_pass_step & = delete;

  static auto record(vkexec::context & /*ctx*/, VkCommandBuffer /*cmd*/, vkexec::detail::pass_cleanup & /*cleanup*/)
    -> vkexec::status
  { return {}; }
};

struct counting_pass_step
{
  int *record_count{ nullptr };
  bool should_fail{ false };

  auto record(vkexec::context & /*ctx*/, VkCommandBuffer /*cmd*/, vkexec::detail::pass_cleanup & /*cleanup*/) const
    -> vkexec::status
  {
    ++(*record_count);
    if (should_fail) { return vkexec::fail(vkexec::errc::invalid_argument); }
    return {};
  }
};

template<class AfterGpu> [[nodiscard]] auto make_empty_pass_step(AfterGpu after_gpu)
{
  return vkexec::detail::make_callback_pass_step(
    [](vkexec::context & /*host*/,
      VkCommandBuffer /*cmd*/,
      vkexec::detail::pass_cleanup & /*cleanup*/) -> vkexec::status { return {}; },
    std::move(after_gpu));
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

using compute_step_t = decltype(vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{}));
using pass_adaptor_t = vkexec::pass_adaptor_sender<vkexec::schedule_sender, compute_step_t>;
using no_push_bind_step_t = vkexec::bind_resources_step<vkexec::detail::no_push_constants>;
using schema_closure_t = vkexec::schema_pass_closure<no_push_bind_step_t, compute_step_t>;
using schema_pass_adaptor_t = vkexec::schema_pass_sender<vkexec::schedule_sender, schema_closure_t>;
using tensor_closure_t = vkexec::tensor_pass_closure<compute_step_t, float>;
using tensor_pass_adaptor_t = vkexec::tensor_pass_sender<vkexec::schedule_sender, tensor_closure_t>;
using move_only_graph_t = vkexec::pass_graph_sender<move_only_pass_step>;

template<class Graph>
concept rvalue_connectable_graph =
  requires(Graph graph, completion_probe_receiver receiver) { ex::connect(std::move(graph), std::move(receiver)); };

template<class Graph>
concept const_lvalue_connectable_graph =
  requires(Graph const &graph, completion_probe_receiver receiver) { ex::connect(graph, std::move(receiver)); };

static_assert(vkexec::detail::static_pass_step<move_only_pass_step>);
static_assert(std::move_constructible<move_only_graph_t>);
static_assert(!std::copy_constructible<move_only_graph_t>);
static_assert(rvalue_connectable_graph<move_only_graph_t>);
static_assert(!const_lvalue_connectable_graph<move_only_graph_t>);
static_assert(
  std::same_as<compute_step_t, vkexec::detail::compute_pass_step<vkexec::detail::no_push_constants, vkexec::dispatch>>);

static_assert(advertises_completion_scheduler<vkexec::schedule_sender, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::schedule_sender, ex::set_error_t>);

static_assert(advertises_completion_scheduler<vkexec::pass_graph_sender<>, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::pass_graph_sender<>, ex::set_error_t>);

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
static_assert(vkexec::vkexec_predecessor<vkexec::pass_graph_sender<>>);
static_assert(std::move_constructible<vkexec::dynamic_pass_graph_sender>);
static_assert(!std::copy_constructible<vkexec::dynamic_pass_graph_sender>);
static_assert(vkexec::detail::is_pass_graph_sender_v<vkexec::dynamic_pass_graph_sender>);
static_assert(vkexec::vkexec_predecessor<vkexec::dynamic_pass_graph_sender>);
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
  auto const step = vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 });
  pass_adaptor_t const sndr{
    .pred = sched.schedule(),
    .step = step,
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

  auto step = make_empty_pass_step([&]() noexcept -> void {
    signal_promise(after_gpu_entered);
    allow_completion_future.wait();
  });
  vkexec::pass_graph_sender<decltype(step)> graph{
    .ctx = ctx.get(),
    .steps = { step },
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
TEST_CASE("dynamic pass graph start returns before GPU completion", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();

  std::promise<void> after_gpu_entered;
  auto after_gpu_ready = after_gpu_entered.get_future();
  std::promise<void> allow_completion;
  auto allow_completion_future = allow_completion.get_future();
  release_promise release_gate{ allow_completion };
  std::promise<void> receiver_completed;
  auto receiver_done = receiver_completed.get_future();

  auto graph =
    vkexec::make_dynamic_pass_graph(ex::schedule(ctx->get_scheduler())) | make_empty_pass_step([&]() noexcept -> void {
      signal_promise(after_gpu_entered);
      allow_completion_future.wait();
    });

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

TEST_CASE("dynamic pass graph executes multiple runtime steps", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();

  int first_recorded = 0;
  int second_recorded = 0;
  int first_after_gpu = 0;
  int second_after_gpu = 0;

  auto first = vkexec::detail::make_callback_pass_step(
    [&](vkexec::context & /*unused*/,
      VkCommandBuffer /*unused*/,
      vkexec::detail::pass_cleanup & /*unused*/) -> vkexec::status {
      ++first_recorded;
      return {};
    },
    [&]() -> void { ++first_after_gpu; });

  auto second = vkexec::detail::make_callback_pass_step(
    [&](vkexec::context & /*unused*/,
      VkCommandBuffer /*unused*/,
      vkexec::detail::pass_cleanup & /*unused*/) -> vkexec::status {
      ++second_recorded;
      return {};
    },
    [&]() -> void { ++second_after_gpu; });

  auto graph = vkexec::make_dynamic_pass_graph(ex::schedule(ctx->get_scheduler())) | first
               | vkexec::barrier::compute_to_compute() | second;

  auto result = vkexec::test::sync_wait_sender(std::move(graph));

  REQUIRE(vkexec::test::sync_wait_completed(result));
  REQUIRE(first_recorded == 1);
  REQUIRE(second_recorded == 1);
  REQUIRE(first_after_gpu == 1);
  REQUIRE(second_after_gpu == 1);
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

  auto step = make_empty_pass_step([&]() noexcept -> void {
    signal_promise(after_gpu_entered);
    allow_completion_future.wait();
  });
  vkexec::pass_graph_sender<decltype(step)> graph{
    .ctx = ctx.get(),
    .steps = { step },
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
  auto step = make_empty_pass_step([]() noexcept -> void {});
  vkexec::pass_graph_sender<decltype(step)> graph{
    .ctx = ctx.get(),
    .steps = { step },
  };

  auto waited = vkexec::test::sync_wait_sender(
    std::move(graph) | ex::then([&]() noexcept -> void { completed_on = std::this_thread::get_id(); }));
  REQUIRE(vkexec::test::sync_wait_completed(waited));
  REQUIRE(completed_on == agent);
}

TEST_CASE("pass composition retains each concrete step type", "[vkexec][pass]")
{
  vkexec::scheduler sched{ nullptr };

  auto graph = ex::schedule(sched) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 });
  STATIC_REQUIRE(vkexec::detail::is_pass_graph_sender_v<decltype(graph)>);
  STATIC_REQUIRE(std::tuple_size_v<decltype(graph.steps)> == 1);

  auto graph2 = std::move(graph) | vkexec::barrier::compute_to_compute();
  STATIC_REQUIRE(vkexec::detail::is_pass_graph_sender_v<decltype(graph2)>);
  STATIC_REQUIRE(std::tuple_size_v<decltype(graph2.steps)> == 2);
  STATIC_REQUIRE(!std::same_as<decltype(graph), decltype(graph2)>);

  auto graph3 = std::move(graph2) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 });
  STATIC_REQUIRE(vkexec::detail::is_pass_graph_sender_v<decltype(graph3)>);
  STATIC_REQUIRE(std::tuple_size_v<decltype(graph3.steps)> == 3);
  STATIC_REQUIRE(!std::same_as<decltype(graph2), decltype(graph3)>);
  REQUIRE(graph3.ctx == nullptr);
}

TEST_CASE("dynamic pass graph keeps one sender type", "[vkexec][pass]")
{
  vkexec::scheduler sched{ nullptr };
  auto graph = vkexec::make_dynamic_pass_graph(ex::schedule(sched));

  graph = std::move(graph) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 });
  STATIC_REQUIRE(std::same_as<decltype(graph), vkexec::dynamic_pass_graph_sender>);
  REQUIRE(graph.steps.size() == 1);

  graph = std::move(graph) | vkexec::barrier::compute_to_compute();
  REQUIRE(graph.steps.size() == 2);

  graph = std::move(graph) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 });
  REQUIRE(graph.steps.size() == 3);
}

TEST_CASE("dynamic pass graph accepts move-only steps", "[vkexec][pass]")
{
  constexpr int k_initial_value = 42;
  vkexec::scheduler sched{ nullptr };
  auto graph = vkexec::make_dynamic_pass_graph(ex::schedule(sched));
  graph = std::move(graph) | move_only_pass_step{ k_initial_value };
  REQUIRE(graph.steps.size() == 1);
}

TEST_CASE("copyable binding graphs defer mutable state to each operation", "[vkexec][pass]")
{
  vkexec::handles::compute_pipeline const pipe{};
  auto graph = ex::schedule(vkexec::scheduler{ nullptr }) | vkexec::bind_resources(pipe, vkexec::resource_table{});

  std::promise<void> first_completed;
  std::promise<void> second_completed;
  auto first = ex::connect(graph, completion_probe_receiver{ .completed = &first_completed });
  auto second = ex::connect(graph, completion_probe_receiver{ .completed = &second_completed });

  REQUIRE(std::get<0>(graph.steps).state == nullptr);
  REQUIRE(std::get<0>(first.steps).state == nullptr);
  REQUIRE(std::get<0>(second.steps).state == nullptr);
}

TEST_CASE("static pass recording stops at the first failure", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  int first_count = 0;
  int failing_count = 0;
  int skipped_count = 0;
  std::tuple steps{
    counting_pass_step{ .record_count = &first_count, .should_fail = false },
    counting_pass_step{ .record_count = &failing_count, .should_fail = true },
    counting_pass_step{ .record_count = &skipped_count, .should_fail = false },
  };
  vkexec::detail::pass_cleanup cleanup{};

  auto recorded = vkexec::detail::record_static_steps(*ctx, VK_NULL_HANDLE, cleanup, steps);

  REQUIRE_FALSE(recorded);
  REQUIRE(first_count == 1);
  REQUIRE(failing_count == 1);
  REQUIRE(skipped_count == 0);
}

TEST_CASE("dynamic pass recording stops at first failure", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  int first_count = 0;
  int failing_count = 0;
  int skipped_count = 0;

  auto graph = vkexec::make_dynamic_pass_graph(ex::schedule(ctx->get_scheduler()));
  graph = std::move(graph) | counting_pass_step{ .record_count = &first_count, .should_fail = false };
  graph = std::move(graph) | counting_pass_step{ .record_count = &failing_count, .should_fail = true };
  graph = std::move(graph) | counting_pass_step{ .record_count = &skipped_count, .should_fail = false };

  vkexec::detail::pass_cleanup cleanup{};
  auto recorded = vkexec::detail::record_dynamic_steps(*ctx, VK_NULL_HANDLE, cleanup, graph.steps);

  REQUIRE_FALSE(recorded);
  REQUIRE(first_count == 1);
  REQUIRE(failing_count == 1);
  REQUIRE(skipped_count == 0);
}
