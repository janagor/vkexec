// NOLINTBEGIN(readability-function-cognitive-complexity)

#include "scheduler_test_support.hpp"
#include "test_helpers.hpp"
#include "vkexec/bind_resources.hpp"
#include "vkexec/detail/pass_execution.hpp"
#include "vkexec/detail/pass_steps.hpp"
#include "vkexec/detail/pass_types.hpp"
#include "vkexec/error.hpp"
#include "vkexec/pass.hpp"
#include "vkexec/pipeline.hpp"
#include "vkexec/queue_submit.hpp"
#include "vkexec/scheduler.hpp"
#include "vkexec/submit_scope.hpp"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <future>
#include <stdexec/execution.hpp>
#include <stdexec/stop_token.hpp>
#include <tuple>
#include <utility>
#include <vector>
#include <vulkan/vulkan_core.h>

TEST_CASE("queue affinity survives static graph composition", "[vkexec][pass]")
{
  constexpr std::uint32_t k_queue_family = 2U;
  vkexec::scheduler sched{ nullptr };
  auto const queue = vkexec::queue_ref{ .queue = VK_NULL_HANDLE, .family = k_queue_family };

  auto first = stdexec::transform_sender(stdexec::schedule(sched) | vkexec::on_queue(queue)
                                           | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{}),
    stdexec::env<>{});
  auto second = stdexec::transform_sender(
    std::move(first) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{}), stdexec::env<>{});

  STATIC_REQUIRE(std::tuple_size_v<decltype(second.steps)> == 2);
  REQUIRE(second.step_queues.size() == std::tuple_size_v<decltype(second.steps)>);
  REQUIRE(second.step_queues.at(0).has_value());
  REQUIRE(second.step_queues.at(1).has_value());
  REQUIRE(second.step_queues.at(0)->family == queue.family);
  REQUIRE(second.step_queues.at(1)->family == queue.family);
}

TEST_CASE("on_queue marks subsequent static pass steps", "[vkexec][pass]")
{
  constexpr std::uint32_t k_graphics_family = 1U;
  constexpr std::uint32_t k_compute_family = 2U;
  vkexec::scheduler sched{ nullptr };
  auto const graphics = vkexec::queue_ref{ .queue = VK_NULL_HANDLE, .family = k_graphics_family };
  auto const compute = vkexec::queue_ref{ .queue = VK_NULL_HANDLE, .family = k_compute_family };

  auto graph = stdexec::transform_sender(stdexec::schedule(sched) | vkexec::on_queue(graphics)
                                           | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{})
                                           | vkexec::on_queue(compute)
                                           | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{}),
    stdexec::env<>{});

  STATIC_REQUIRE(std::tuple_size_v<decltype(graph.steps)> == 2);
  REQUIRE(graph.step_queues.size() == std::tuple_size_v<decltype(graph.steps)>);
  REQUIRE(graph.step_queues.at(0).has_value());
  REQUIRE(graph.step_queues.at(1).has_value());
  REQUIRE(graph.step_queues.at(0)->family == graphics.family);
  REQUIRE(graph.step_queues.at(1)->family == compute.family);
  REQUIRE(graph.current_queue.has_value());
  REQUIRE(graph.current_queue->family == compute.family);
}

TEST_CASE("dynamic pass graph accepts move-only steps", "[vkexec][pass]")
{
  constexpr int k_initial_value = 42;
  vkexec::scheduler sched{ nullptr };
  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(sched));
  graph.append(vkexec::make_pass_adaptor(move_only_pass_step{ k_initial_value }));
  REQUIRE(graph.steps.size() == 1);
}

TEST_CASE("pass recording forwards queue and batch identity through static and dynamic steps", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto const queue = ctx->compute_queue_ref();
  auto const info = vkexec::detail::recording_info(queue);
  auto const other_info = vkexec::detail::recording_info(queue);
  REQUIRE(info.submission_id == 0);
  REQUIRE(other_info.submission_id == 0);
  vkexec::pass_recording_info static_observed{};
  vkexec::pass_recording_info dynamic_observed{};
  auto static_steps = std::tuple{ informed_pass_step{ .observed = &static_observed } };
  std::vector<vkexec::detail::dynamic_pass_step> dynamic_steps;
  dynamic_steps.emplace_back(informed_pass_step{ .observed = &dynamic_observed });
  vkexec::detail::pass_cleanup cleanup{};

  REQUIRE(vkexec::detail::record_static_steps(*ctx, VK_NULL_HANDLE, cleanup, static_steps, info));
  REQUIRE(vkexec::detail::record_dynamic_steps(*ctx, VK_NULL_HANDLE, cleanup, dynamic_steps, info));
  REQUIRE(static_observed.queue.queue == queue.queue);
  REQUIRE(dynamic_observed.queue.queue == queue.queue);
  REQUIRE(static_observed.queue.family == queue.family);
  REQUIRE(dynamic_observed.queue.family == queue.family);
  REQUIRE(static_observed.submission_id == info.submission_id);
  REQUIRE(dynamic_observed.submission_id == info.submission_id);
  REQUIRE(info.ensure_submission_id() != other_info.ensure_submission_id());
}

TEST_CASE("ordinary pass completion commit runs only after success", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  int commits = 0;
  auto successful =
    stdexec::schedule(ctx->get_scheduler()) | vkexec::make_pass_adaptor(completion_commit_step{ .commits = &commits });
  auto success = vkexec::test::sync_wait_sender(std::move(successful));
  REQUIRE(vkexec::test::sync_wait_completed(success));
  REQUIRE(commits == 1);

  auto failing = stdexec::schedule(ctx->get_scheduler())
                 | vkexec::make_pass_adaptor(completion_commit_step{ .commits = &commits, .should_fail = true });
  auto failure = vkexec::test::sync_wait_sender(std::move(failing));
  REQUIRE(failure.failed());
  REQUIRE(commits == 1);

  stdexec::inplace_stop_source source;
  source.request_stop();
  auto cancelled = stdexec::write_env(
    stdexec::schedule(ctx->get_scheduler()) | vkexec::make_pass_adaptor(completion_commit_step{ .commits = &commits }),
    stdexec::prop{ stdexec::get_stop_token, source.get_token() });
  auto stopped = vkexec::test::sync_wait_sender(std::move(cancelled));
  REQUIRE(vkexec::test::sync_wait_stopped(stopped));
  REQUIRE(commits == 1);
}

TEST_CASE("pass recording prefers the context-aware overload", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  int selected{};
  auto step = dual_record_pass_step{ .selected = &selected };
  vkexec::detail::pass_cleanup cleanup{};
  auto const info = vkexec::detail::recording_info(ctx->compute_queue_ref());

  REQUIRE(vkexec::detail::record_step(step, *ctx, VK_NULL_HANDLE, cleanup, info));
  REQUIRE(selected == dual_record_pass_step::k_context_signature);
}

TEST_CASE("static graph shares recording identity within a batch and changes it on replay", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto const queue = ctx->compute_queue_ref();
  vkexec::pass_recording_info first{};
  vkexec::pass_recording_info second{};
  auto make_graph = [&]() -> decltype(auto) {
    return stdexec::schedule(ctx->get_scheduler()) | vkexec::make_pass_adaptor(informed_pass_step{ .observed = &first })
           | vkexec::make_pass_adaptor(informed_pass_step{ .observed = &second });
  };

  REQUIRE(vkexec::test::sync_wait_completed(vkexec::test::sync_wait_sender(make_graph())));
  REQUIRE(first.submission_id >= vkexec::detail::k_first_submission_id);
  REQUIRE(first.submission_id == second.submission_id);
  REQUIRE(first.queue.queue == queue.queue);
  REQUIRE(first.queue.family == queue.family);

  auto const previous_id = first.submission_id;
  REQUIRE(vkexec::test::sync_wait_completed(vkexec::test::sync_wait_sender(make_graph())));
  REQUIRE(first.submission_id != previous_id);
  REQUIRE(first.submission_id == second.submission_id);
}

TEST_CASE("dynamic graph forwards batch identity and explicit queue affinity", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto const compute = ctx->compute_queue_ref();
  auto const graphics = ctx->graphics_queue_ref();
  auto const other = graphics.queue == VK_NULL_HANDLE ? compute : graphics;
  vkexec::pass_recording_info first{};
  vkexec::pass_recording_info second{};
  vkexec::pass_recording_info third{};
  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  graph.append_queue(compute);
  graph.append(vkexec::make_pass_adaptor(informed_pass_step{ .observed = &first }));
  graph.append(vkexec::make_pass_adaptor(informed_pass_step{ .observed = &second }));
  graph.append_queue(other);
  graph.append(vkexec::make_pass_adaptor(informed_pass_step{ .observed = &third }));

  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
  REQUIRE(first.submission_id >= vkexec::detail::k_first_submission_id);
  REQUIRE(first.submission_id == second.submission_id);
  if (vkexec::detail::same_queue(compute, other)) {
    REQUIRE(first.submission_id == third.submission_id);
  } else {
    REQUIRE(first.submission_id != third.submission_id);
  }
  REQUIRE(first.queue.queue == compute.queue);
  REQUIRE(first.queue.family == compute.family);
  REQUIRE(third.queue.queue == other.queue);
  REQUIRE(third.queue.family == other.family);
}

TEST_CASE("independent DAG nodes receive distinct recording identities", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  vkexec::pass_recording_info first{};
  vkexec::pass_recording_info second{};
  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  graph.append_queue(ctx->compute_queue_ref());
  (void)graph.append_after({}, vkexec::make_pass_adaptor(informed_pass_step{ .observed = &first }));
  (void)graph.append_after({}, vkexec::make_pass_adaptor(informed_pass_step{ .observed = &second }));

  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
  REQUIRE(first.submission_id >= vkexec::detail::k_first_submission_id);
  REQUIRE(second.submission_id >= vkexec::detail::k_first_submission_id);
  REQUIRE(first.submission_id != second.submission_id);
}

TEST_CASE("four-argument recording failure propagates from a graph", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  vkexec::pass_recording_info observed{};
  int skipped{};
  auto graph = stdexec::schedule(ctx->get_scheduler())
               | vkexec::make_pass_adaptor(informed_pass_step{ .observed = &observed, .should_fail = true })
               | vkexec::make_pass_adaptor(counting_pass_step{ .record_count = &skipped });

  auto const result = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(result.failed());
  REQUIRE(result.error.has_value());
  REQUIRE(result.error.value().code == vkexec::make_error_code(vkexec::errc::invalid_argument));
  REQUIRE_FALSE(result.stopped);
  REQUIRE(observed.submission_id >= vkexec::detail::k_first_submission_id);
  REQUIRE(skipped == 0);
}

TEST_CASE("copyable binding graphs defer mutable state to each operation", "[vkexec][pass]")
{
  vkexec::handles::compute_pipeline const pipe{};
  auto graph = stdexec::transform_sender(
    stdexec::schedule(vkexec::scheduler{ nullptr }) | vkexec::bind_resources(pipe, vkexec::resource_table{}),
    stdexec::env<>{});

  std::promise<void> first_completed;
  std::promise<void> second_completed;
  auto first = graph.connect(completion_probe_receiver{ .completed = &first_completed });
  auto second = graph.connect(completion_probe_receiver{ .completed = &second_completed });

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

  auto recorded = vkexec::detail::record_static_steps(
    *ctx, VK_NULL_HANDLE, cleanup, steps, vkexec::detail::recording_info(ctx->compute_queue_ref()));

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

  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  graph.append(vkexec::make_pass_adaptor(counting_pass_step{ .record_count = &first_count, .should_fail = false }));
  graph.append(vkexec::make_pass_adaptor(counting_pass_step{ .record_count = &failing_count, .should_fail = true }));
  graph.append(vkexec::make_pass_adaptor(counting_pass_step{ .record_count = &skipped_count, .should_fail = false }));

  vkexec::detail::pass_cleanup cleanup{};
  auto recorded = vkexec::detail::record_dynamic_steps(
    *ctx, VK_NULL_HANDLE, cleanup, graph.steps, vkexec::detail::recording_info(ctx->compute_queue_ref()));

  REQUIRE_FALSE(recorded);
  REQUIRE(first_count == 1);
  REQUIRE(failing_count == 1);
  REQUIRE(skipped_count == 0);
}


TEST_CASE("DAG dependencies complete across distinct queue handles", "[vkexec][pass][gpu][sync]")
{
  auto ctx = vkexec::test::require_context();
  auto const compute = ctx->compute_queue_ref();
  auto const graphics = ctx->graphics_queue_ref();
  if (graphics.queue == VK_NULL_HANDLE || graphics.queue == compute.queue) {
    SKIP("default context has no distinct graphics queue; multi-queue GPU coverage unavailable");
  }

  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  graph.append_queue(compute);
  auto const source =
    graph.append_after({}, vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {}));
  std::array const source_dependency{ source };
  graph.append_queue(graphics);
  auto const first_branch =
    graph.append_after(source_dependency, vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {}));
  auto const second_branch =
    graph.append_after(source_dependency, vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {}));
  std::array const branch_dependencies{ first_branch, second_branch };
  graph.append_queue(compute);
  (void)graph.append_after(
    branch_dependencies, vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {}));

  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
}

// NOLINTEND(readability-function-cognitive-complexity)
