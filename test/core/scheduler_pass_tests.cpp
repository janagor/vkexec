// NOLINTBEGIN(readability-function-cognitive-complexity)

#include <array>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <span>
#include <stdexec/execution.hpp>
#include <thread>
#include <tuple>// IWYU pragma: keep
#include <type_traits>// IWYU pragma: keep
#include <utility>
#include <vector>// IWYU pragma: keep
#include <vulkan/vulkan_core.h>

#include "scheduler_test_support.hpp"
#include "test_helpers.hpp"
#include "vkexec/barrier.hpp"
#include "vkexec/context.hpp"
#include "vkexec/detail/pass_execution.hpp"
#include "vkexec/detail/pass_steps.hpp"
#include "vkexec/detail/pass_types.hpp"
#include "vkexec/detail/sender_expr.hpp"
#include "vkexec/pass.hpp"
#include "vkexec/queue_submit.hpp"// IWYU pragma: keep
#include "vkexec/resource_use.hpp"// IWYU pragma: keep
#include "vkexec/result.hpp"
#include "vkexec/scheduler.hpp"
#include "vkexec/submit_scope.hpp"

namespace vkexec {
struct domain;
}// namespace vkexec

TEST_CASE("schedule_sender advertises completion domain", "[vkexec][scheduler][domain]")
{
  vkexec::scheduler_env const env{};
  STATIC_REQUIRE(
    std::same_as<decltype(env.query(stdexec::get_completion_domain_t<stdexec::set_value_t>{})), vkexec::domain>);
  STATIC_REQUIRE(std::same_as<decltype(env.query(stdexec::get_domain_t{})), vkexec::domain>);

  vkexec::scheduler const sched{ nullptr };
  STATIC_REQUIRE(
    std::same_as<decltype(sched.query(stdexec::get_completion_domain_t<stdexec::set_value_t>{})), vkexec::domain>);
  STATIC_REQUIRE(std::same_as<decltype(sched.query(stdexec::get_domain_t{})), vkexec::domain>);
}

TEST_CASE("domain-only sender environment advertises vkexec domain", "[vkexec][scheduler][domain]")
{
  vkexec::detail::domain_env const env{};

  STATIC_REQUIRE(
    std::same_as<decltype(env.query(stdexec::get_completion_domain_t<stdexec::set_value_t>{})), vkexec::domain>);
  STATIC_REQUIRE(std::same_as<decltype(env.query(stdexec::get_domain_t{})), vkexec::domain>);
}

TEST_CASE("semantic pass expression preserves predecessor value completion scheduler", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto pred = sched.schedule() | stdexec::then([]() -> void {});
  auto const sndr = pred | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 });

  STATIC_REQUIRE(stdexec::sender<decltype(sndr)>);
  auto const completion = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr));
  REQUIRE(completion == sched);
}

TEST_CASE("pass closures compose independently from a sender", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto pipeline = vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 })
                  | vkexec::barrier::compute_to_compute()
                  | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 });
  auto sndr = sched.schedule() | pipeline;
  auto second = sched.schedule() | pipeline;
  STATIC_REQUIRE(stdexec::sender<decltype(sndr)>);
  STATIC_REQUIRE(vkexec::detail::is_sender_expr_v<decltype(sndr)>);
  STATIC_REQUIRE(std::same_as<vkexec::detail::expression_tag_t<decltype(sndr)>, vkexec::compute_pass_t>);
  auto lowered = stdexec::transform_sender(sndr, stdexec::env<>{});
  auto second_lowered = stdexec::transform_sender(second, stdexec::env<>{});
  REQUIRE(!lowered.state);
  REQUIRE(!second_lowered.state);
}

TEST_CASE("compute pass direct and pipe forms produce the same sender", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto direct = vkexec::compute_pass(sched.schedule(), vkexec::compute_bind{}, vkexec::dispatch{});
  auto piped = sched.schedule() | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{});

  STATIC_REQUIRE(std::same_as<decltype(direct), decltype(piped)>);
  STATIC_REQUIRE(vkexec::detail::is_sender_expr_v<decltype(direct)>);
  REQUIRE(stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(direct))
          == stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(piped)));
}

TEST_CASE("semantic pass expressions fuse into one typed graph", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto expr = sched.schedule() | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{})
              | vkexec::barrier::compute_to_compute()
              | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{});
  auto sndr = stdexec::transform_sender(expr, stdexec::env<>{});

  STATIC_REQUIRE(std::tuple_size_v<std::remove_cvref_t<decltype(sndr.steps)>> == 3);
  REQUIRE(std::get<0>(sndr.steps).bind.pipeline == VK_NULL_HANDLE);
}

TEST_CASE("semantic pass expressions materialize across a then boundary", "[vkexec][scheduler]")
{
  using expr_t = decltype(stdexec::schedule(std::declval<vkexec::scheduler>()) | stdexec::then([]() noexcept -> void {})
                          | vkexec::make_pass_adaptor(std::declval<counting_pass_step>())
                          | vkexec::make_pass_adaptor(std::declval<counting_pass_step>()));

  STATIC_REQUIRE(stdexec::sender<expr_t>);
  using lowered_t = decltype(stdexec::transform_sender(std::declval<expr_t>(), stdexec::env<>{}));
  STATIC_REQUIRE(stdexec::sender<lowered_t>);
}

TEST_CASE("pass_graph_sender start returns before GPU completion", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();

  std::promise<void> after_gpu_entered;
  auto after_gpu_ready = after_gpu_entered.get_future();
  std::atomic_bool allow_completion{ false };
  release_flag release_gate{ allow_completion };
  std::promise<void> receiver_completed;
  auto receiver_done = receiver_completed.get_future();

  auto step = make_empty_pass_step([&]() noexcept -> void {
    signal_promise(after_gpu_entered);
    allow_completion.wait(false, std::memory_order_acquire);
  });
  vkexec::pass_graph_sender<decltype(step)> graph{
    .state = vkexec::detail::context_access::state(*ctx),
    .steps = { step },
    .step_queues = {},
    .step_predecessors = {},
    .presentation = {},
    .current_queue = {},
  };

  auto operation = stdexec::connect(graph, completion_probe_receiver{ .completed = &receiver_completed });

  std::promise<void> start_returned;
  auto start_done = start_returned.get_future();

  std::thread starter{ [&]() -> void {
    stdexec::start(operation);
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

TEST_CASE("dynamic pass graph start returns before GPU completion", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();

  std::promise<void> after_gpu_entered;
  auto after_gpu_ready = after_gpu_entered.get_future();
  std::atomic_bool allow_completion{ false };
  release_flag release_gate{ allow_completion };
  std::promise<void> receiver_completed;
  auto receiver_done = receiver_completed.get_future();

  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()))
               | vkexec::make_pass_adaptor(make_empty_pass_step([&]() noexcept -> void {
                   signal_promise(after_gpu_entered);
                   allow_completion.wait(false, std::memory_order_acquire);
                 }));

  auto operation = stdexec::connect(std::move(graph), completion_probe_receiver{ .completed = &receiver_completed });

  std::promise<void> start_returned;
  auto start_done = start_returned.get_future();

  std::thread starter{ [&]() -> void {
    stdexec::start(operation);
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

  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()))
               | vkexec::make_pass_adaptor(first) | vkexec::barrier::compute_to_compute()
               | vkexec::make_pass_adaptor(second);

  auto result = vkexec::test::sync_wait_sender(std::move(graph));

  REQUIRE(vkexec::test::sync_wait_completed(result));
  REQUIRE(first_recorded == 1);
  REQUIRE(second_recorded == 1);
  REQUIRE(first_after_gpu == 1);
  REQUIRE(second_after_gpu == 1);
}

TEST_CASE("DAG fan-out and fan-in completes all submitted branches", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  auto const compute = ctx->compute_queue_ref();
  auto const graphics = ctx->graphics_queue_ref();
  auto make_empty_node = []() -> decltype(auto) {
    return vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*unused*/) -> void {});
  };

  graph.append_queue(compute);
  auto const root = graph.append_after({}, make_empty_node());
  std::array const root_dependency{ root };
  auto const left = graph.append_after(root_dependency, make_empty_node());
  if (graphics.queue != VK_NULL_HANDLE) { graph.append_queue(graphics); }
  auto const right = graph.append_after(root_dependency, make_empty_node());
  std::array const join_dependencies{ left, right };
  graph.append_queue(compute);
  (void)graph.append_after(join_dependencies, make_empty_node());

  auto const result = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(result));
}

TEST_CASE("DAG terminal join waits for independent queue branches", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto const compute = ctx->compute_queue_ref();
  auto const graphics = ctx->graphics_queue_ref();
  if (graphics.queue == VK_NULL_HANDLE || vkexec::detail::same_queue(compute, graphics)) {
    SKIP("Distinct graphics and compute queues are unavailable");
  }

  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  graph.append_queue(compute);
  auto const compute_branch =
    graph.append_after({}, vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*unused*/) -> void {}));
  graph.append_queue(graphics);
  auto const graphics_branch =
    graph.append_after({}, vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*unused*/) -> void {}));
  REQUIRE(compute_branch != graphics_branch);

  auto const result = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(result));
}

TEST_CASE("typed DAG executes branches and their join", "[vkexec][pass][gpu]")
{
  constexpr int k_recorded_once = 1;
  auto ctx = vkexec::test::require_context();
  auto const compute = ctx->compute_queue_ref();
  auto const graphics = ctx->graphics_queue_ref();
  auto const right_queue = graphics.queue == VK_NULL_HANDLE ? compute : graphics;

  int root_recorded{};
  int left_recorded{};
  int right_recorded{};
  int joined_recorded{};
  auto make_step = [](int &count) -> decltype(auto) {
    return vkexec::custom_pass(vkexec::uses(), [&count](VkCommandBuffer /*unused*/) -> void { ++count; });
  };
  auto graph = stdexec::schedule(ctx->get_scheduler()) | vkexec::on_queue(compute) | make_step(root_recorded)
               | vkexec::when_all(vkexec::on_queue(compute) | make_step(left_recorded),
                 vkexec::on_queue(right_queue) | make_step(right_recorded))
               | make_step(joined_recorded);

  auto const result = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(result));
  REQUIRE(root_recorded == k_recorded_once);
  REQUIRE(left_recorded == k_recorded_once);
  REQUIRE(right_recorded == k_recorded_once);
  REQUIRE(joined_recorded == k_recorded_once);
}

TEST_CASE("sync_wait still waits for pass_graph_sender completion", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();

  std::promise<void> after_gpu_entered;
  auto after_gpu_ready = after_gpu_entered.get_future();
  std::atomic_bool allow_completion{ false };
  release_flag release_gate{ allow_completion };
  std::atomic_bool sync_wait_returned{ false };
  std::atomic_bool completed_successfully{ false };

  auto step = make_empty_pass_step([&]() noexcept -> void {
    signal_promise(after_gpu_entered);
    allow_completion.wait(false, std::memory_order_acquire);
  });
  vkexec::pass_graph_sender<decltype(step)> graph{
    .state = vkexec::detail::context_access::state(*ctx),
    .steps = { step },
    .step_queues = {},
    .step_predecessors = {},
    .presentation = {},
    .current_queue = {},
  };

  std::thread waiter{ [&]() -> void {
    auto const waited = vkexec::test::sync_wait_sender(graph);
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
    .state = vkexec::detail::context_access::state(*ctx),
    .steps = { step },
    .step_queues = {},
    .step_predecessors = {},
    .presentation = {},
    .current_queue = {},
  };

  auto waited = vkexec::test::sync_wait_sender(
    std::move(graph) | stdexec::then([&]() noexcept -> void { completed_on = std::this_thread::get_id(); }));
  REQUIRE(vkexec::test::sync_wait_completed(waited));
  REQUIRE(completed_on == agent);
}

TEST_CASE("pass composition retains each concrete step type", "[vkexec][pass]")
{
  vkexec::scheduler const sched{ nullptr };

  auto graph = stdexec::transform_sender(
    stdexec::schedule(sched) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 }),
    stdexec::env<>{});
  STATIC_REQUIRE(vkexec::detail::is_pass_graph_sender_v<decltype(graph)>);
  STATIC_REQUIRE(std::tuple_size_v<decltype(graph.steps)> == 1);

  auto graph2 = stdexec::transform_sender(std::move(graph) | vkexec::barrier::compute_to_compute(), stdexec::env<>{});
  STATIC_REQUIRE(vkexec::detail::is_pass_graph_sender_v<decltype(graph2)>);
  STATIC_REQUIRE(std::tuple_size_v<decltype(graph2.steps)> == 2);
  STATIC_REQUIRE(!std::same_as<decltype(graph), decltype(graph2)>);

  auto graph3 = stdexec::transform_sender(
    std::move(graph2) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 }), stdexec::env<>{});
  STATIC_REQUIRE(vkexec::detail::is_pass_graph_sender_v<decltype(graph3)>);
  STATIC_REQUIRE(std::tuple_size_v<decltype(graph3.steps)> == 3);
  STATIC_REQUIRE(!std::same_as<decltype(graph2), decltype(graph3)>);
  REQUIRE(!graph3.state);
}

TEST_CASE("typed graph appended after materialization joins branch sinks", "[vkexec][pass]")
{
  constexpr std::size_t k_root = 0;
  constexpr std::size_t k_left = 1;
  constexpr std::size_t k_right = 2;
  constexpr std::size_t k_joined = 3;
  constexpr std::size_t k_step_count = 4;
  vkexec::scheduler const sched{ nullptr };
  auto make_step = []() -> decltype(auto) {
    return vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*unused*/) -> void {});
  };

  auto branched = stdexec::transform_sender(
    stdexec::schedule(sched) | make_step() | vkexec::when_all(make_step(), make_step()), stdexec::env<>{});
  STATIC_REQUIRE(vkexec::detail::is_pass_graph_sender_v<decltype(branched)>);
  REQUIRE(branched.step_predecessors == std::vector<std::vector<std::size_t>>{ {}, { k_root }, { k_root } });

  auto graph = stdexec::transform_sender(std::move(branched) | make_step(), stdexec::env<>{});
  STATIC_REQUIRE(vkexec::detail::is_pass_graph_sender_v<decltype(graph)>);
  STATIC_REQUIRE(std::tuple_size_v<decltype(graph.steps)> == k_step_count);
  REQUIRE(graph.step_predecessors.at(k_joined) == std::vector<std::size_t>{ k_left, k_right });
}

TEST_CASE("dynamic pass graph keeps one sender type", "[vkexec][pass]")
{
  constexpr std::uint32_t k_queue_family = 2U;
  constexpr std::uint32_t k_next_queue_family = 3U;
  vkexec::scheduler const sched{ nullptr };
  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(sched));
  auto const queue = vkexec::queue_ref{ .queue = VK_NULL_HANDLE, .family = k_queue_family };
  auto const next_queue = vkexec::queue_ref{ .queue = VK_NULL_HANDLE, .family = k_next_queue_family };

  graph.append(vkexec::on_queue(queue));
  REQUIRE(graph.steps.empty());

  graph.append(vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 }));
  STATIC_REQUIRE(std::same_as<decltype(graph), vkexec::dynamic_pass_graph_sender>);
  REQUIRE(graph.steps.size() == 1);
  REQUIRE(graph.step_queues.at(0).has_value());
  REQUIRE(graph.step_queues.at(0)->family == queue.family);

  graph.append(vkexec::barrier::compute_to_compute());
  REQUIRE(graph.steps.size() == 2);

  graph.append(vkexec::on_queue(next_queue));
  graph.append(vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 }));
  REQUIRE(graph.steps.size() == 3);
  REQUIRE(graph.step_queues.size() == graph.steps.size());
  REQUIRE(graph.step_queues.at(1).has_value());
  REQUIRE(graph.step_queues.at(1)->family == queue.family);
  REQUIRE(graph.step_queues.at(2).has_value());
  REQUIRE(graph.step_queues.at(2)->family == next_queue.family);
}


// NOLINTEND(readability-function-cognitive-complexity)
