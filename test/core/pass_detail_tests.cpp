// NOLINTBEGIN(readability-function-cognitive-complexity)

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <span>
#include <stdexec/execution.hpp>
#include <tuple>
#include <utility>
#include <vector>
#include <vulkan/vulkan_core.h>

#include "pass_detail_test_support.hpp"
#include "vkexec/barrier_params.hpp"
#include "vkexec/detail/pass_execution.hpp"
#include "vkexec/detail/pass_graph_submit.hpp"
#include "vkexec/detail/pass_lowering.hpp"
#include "vkexec/detail/pass_steps.hpp"
#include "vkexec/detail/pass_types.hpp"
#include "vkexec/detail/resource_state_tracker.hpp"
#include "vkexec/detail/resource_state_tracker_helpers.hpp"
#include "vkexec/detail/sender_expr.hpp"
#include "vkexec/pass.hpp"
#include "vkexec/pipeline.hpp"
#include "vkexec/queue_submit.hpp"
#include "vkexec/resource_use.hpp"
#include "vkexec/result.hpp"
#include "vkexec/scheduler.hpp"
#include "vkexec/submit_scope.hpp"

namespace vkexec {
class context;
}// namespace vkexec

TEST_CASE("storage_bindings_equal compares buffer bindings", "[vkexec][pass]")
{
  using vkexec::detail::storage_bindings_equal;

  char buffer_left{};
  char buffer_right{};
  vkexec::storage_binding const left_binding{
    .buffer = fake_buffer(&buffer_left), .byte_size = k_byte_size, .binding = 0
  };
  vkexec::storage_binding const right_binding{
    .buffer = fake_buffer(&buffer_right),
    .byte_size = k_byte_size,
    .binding = 0,
  };
  vkexec::storage_binding const left_copy = left_binding;

  REQUIRE(storage_bindings_equal(std::span{ &left_binding, 1 }, std::span{ &left_copy, 1 }));
  REQUIRE_FALSE(storage_bindings_equal(std::span{ &left_binding, 1 }, std::span{ &right_binding, 1 }));
  REQUIRE(
    storage_bindings_equal(std::span<vkexec::storage_binding const>{}, std::span<vkexec::storage_binding const>{}));

  vkexec::storage_binding const diff_binding{
    .buffer = left_binding.buffer,
    .byte_size = left_binding.byte_size,
    .binding = 1,
  };
  vkexec::storage_binding const diff_size_binding{
    .buffer = left_binding.buffer,
    .byte_size = left_binding.byte_size + 1,
    .binding = left_binding.binding,
  };

  REQUIRE_FALSE(storage_bindings_equal(std::span{ &left_binding, 1 }, std::span{ &diff_size_binding, 1 }));
  REQUIRE_FALSE(storage_bindings_equal(std::span{ &left_binding, 1 }, std::span{ &diff_binding, 1 }));

  std::array const bindings{ left_binding, right_binding };
  std::array const bindings_reordered{ right_binding, left_binding };
  REQUIRE_FALSE(storage_bindings_equal(std::span{ bindings }, std::span{ bindings_reordered }));
  REQUIRE(storage_bindings_equal(std::span{ bindings }, std::span{ bindings }));
}

TEST_CASE("write_storage_descriptors returns when no buffers are bound", "[vkexec][pass]")
{ vkexec::write_storage_descriptors(VK_NULL_HANDLE, VK_NULL_HANDLE, std::span<vkexec::storage_binding const>{}); }

TEST_CASE("pass batches group adjacent steps by resolved queue", "[vkexec][pass][dag-plan]")
{
  char graphics_storage{};
  char compute_storage{};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  vkexec::queue_ref const graphics{ .queue = reinterpret_cast<VkQueue>(&graphics_storage), .family = k_default_family };
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  vkexec::queue_ref const compute{ .queue = reinterpret_cast<VkQueue>(&compute_storage), .family = k_other_family };
  vkexec::queue_affinity const unset{};
  vkexec::queue_affinity const graphics_affinity{ graphics };
  vkexec::queue_affinity const compute_affinity{ compute };

  auto const empty = vkexec::detail::plan_batches({}, graphics);
  REQUIRE(empty.batches.empty());

  std::array const one{ graphics_affinity };
  auto const one_plan = vkexec::detail::plan_batches(one, graphics);
  REQUIRE(one_plan.batches.size() == 1);
  REQUIRE(one_plan.batches.front().first_step == 0);
  REQUIRE(one_plan.batches.front().end_step == 1);

  std::array const repeated{ graphics_affinity, graphics_affinity, graphics_affinity };
  auto const repeated_plan = vkexec::detail::plan_batches(repeated, graphics);
  REQUIRE(repeated_plan.batches.size() == 1);
  REQUIRE(repeated_plan.batches.front().end_step == repeated.size());

  std::array const switch_once{ graphics_affinity, compute_affinity };
  auto const switch_plan = vkexec::detail::plan_batches(switch_once, graphics);
  REQUIRE(switch_plan.batches.size() == 2);
  REQUIRE(switch_plan.batches.front().end_step == 1);
  REQUIRE(switch_plan.batches.back().first_step == 1);

  std::array const switch_twice{
    graphics_affinity, graphics_affinity, compute_affinity, compute_affinity, graphics_affinity
  };
  auto const three_batches = vkexec::detail::plan_batches(switch_twice, graphics);
  REQUIRE(three_batches.batches.size() == 3);
  REQUIRE(three_batches.batches.front().end_step == 2);
  REQUIRE(three_batches.batches.at(1).first_step == 2);
  REQUIRE(three_batches.batches.at(1).end_step == 4);
  REQUIRE(three_batches.batches.back().first_step == 4);
  REQUIRE(three_batches.batches.back().end_step == switch_twice.size());

  std::array const defaults{ unset, unset };
  auto const default_plan = vkexec::detail::plan_batches(defaults, graphics);
  REQUIRE(default_plan.batches.size() == 1);
  REQUIRE(default_plan.batches.front().queue.queue == graphics.queue);
  REQUIRE(default_plan.batches.front().end_step == defaults.size());

  std::array const mixed{ unset, compute_affinity, compute_affinity, unset };
  auto const mixed_plan = vkexec::detail::plan_batches(mixed, graphics);
  REQUIRE(mixed_plan.batches.size() == 3);
  REQUIRE(mixed_plan.batches.front().queue.queue == graphics.queue);
  REQUIRE(mixed_plan.batches.at(1).queue.queue == compute.queue);
  REQUIRE(mixed_plan.batches.at(1).first_step == 1);
  REQUIRE(mixed_plan.batches.at(1).end_step == 3);
  REQUIRE(mixed_plan.batches.back().queue.queue == graphics.queue);
  REQUIRE(mixed_plan.batches.back().first_step == 3);

  auto planned = vkexec::detail::plan_execution(mixed.size(), mixed, graphics);
  REQUIRE(planned.has_value());
  REQUIRE(planned->batches.size() == mixed_plan.batches.size());
  REQUIRE_FALSE(vkexec::detail::plan_execution(mixed.size() - 1, mixed, graphics).has_value());
}

TEST_CASE("DAG planner keeps independent branches separate", "[vkexec][pass][dag-plan]")
{
  char first_storage{};
  char second_storage{};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  vkexec::queue_ref const first{ .queue = reinterpret_cast<VkQueue>(&first_storage), .family = 0 };
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  vkexec::queue_ref const second{ .queue = reinterpret_cast<VkQueue>(&second_storage), .family = 1 };
  std::array<vkexec::queue_affinity, k_dag_step_count> const queues{ first, first, second, first };
  std::array<std::vector<std::size_t>, k_dag_step_count> const predecessors{
    std::vector<std::size_t>{},
    std::vector<std::size_t>{ k_first_step },
    std::vector<std::size_t>{ k_first_step },
    std::vector<std::size_t>{ k_second_step, k_second_step + std::size_t{ 1 } },
  };
  auto planned = vkexec::detail::plan_dag(queues, predecessors, first);
  REQUIRE(planned.has_value());
  REQUIRE(planned->batches.size() == k_dag_step_count);
  REQUIRE(planned->batches.at(k_second_step).predecessors == predecessors.at(k_second_step));
  REQUIRE(planned->batches.at(k_second_step + std::size_t{ 1 }).predecessors
          == predecessors.at(k_second_step + std::size_t{ 1 }));
  REQUIRE(planned->batches.back().predecessors == predecessors.back());
  REQUIRE(
    vkexec::detail::terminal_batches(*planned) == std::vector<std::size_t>{ k_dag_step_count - std::size_t{ 1 } });

  auto invalid = predecessors;
  invalid.at(k_second_step).push_back(k_second_step);
  REQUIRE_FALSE(vkexec::detail::plan_dag(queues, invalid, first).has_value());

  invalid = predecessors;
  invalid.at(k_second_step).push_back(k_first_step);
  REQUIRE_FALSE(vkexec::detail::plan_dag(queues, invalid, first).has_value());

  invalid = predecessors;
  invalid.back().clear();
  auto independent = vkexec::detail::plan_dag(queues, invalid, first);
  REQUIRE(independent.has_value());
  REQUIRE(vkexec::detail::terminal_batches(*independent).size() == k_dag_step_count - std::size_t{ 1 });
}

TEST_CASE("partial graph fence creation retains ownership of internal fences", "[vkexec][pass]")
{
  REQUIRE_FALSE(vkexec::detail::is_borrowed_graph_fence(k_second_step, k_dag_step_count, true));
  REQUIRE(vkexec::detail::is_borrowed_graph_fence(k_dag_step_count - std::size_t{ 1 }, k_dag_step_count, true));
  REQUIRE_FALSE(vkexec::detail::is_borrowed_graph_fence(k_dag_step_count - std::size_t{ 1 }, k_dag_step_count, false));
}

TEST_CASE("dynamic pass graph records branch predecessors", "[vkexec][pass]")
{
  vkexec::dynamic_pass_graph_sender graph{};
  auto make_step = []() -> decltype(auto) {
    return vkexec::make_pass_adaptor(vkexec::detail::make_callback_pass_step(
      [](vkexec::context &, VkCommandBuffer, vkexec::detail::pass_cleanup &) -> vkexec::status { return {}; }));
  };
  auto const root = graph.append_after({}, make_step());
  std::array const root_dependency{ root };
  auto const left = graph.append_after(root_dependency, make_step());
  auto const right = graph.append_after(root_dependency, make_step());
  std::array const join_dependencies{ left, right };
  auto const joined = graph.append_after(join_dependencies, make_step());

  REQUIRE(joined == k_dag_step_count - std::size_t{ 1 });
  REQUIRE(graph.step_predecessors.at(left) == std::vector<std::size_t>{ root });
  REQUIRE(graph.step_predecessors.at(right) == std::vector<std::size_t>{ root });
  REQUIRE(graph.step_predecessors.at(joined).size() == k_pair_step_count);
  REQUIRE(graph.step_predecessors.at(joined).at(k_first_step) == left);
  REQUIRE(graph.step_predecessors.at(joined).at(k_second_step) == right);
}

TEST_CASE("typed branches join at the next pass", "[vkexec][pass]")
{
  constexpr std::size_t k_root = 0;
  constexpr std::size_t k_left_first = 1;
  constexpr std::size_t k_left_last = 2;
  constexpr std::size_t k_right = 3;
  constexpr std::size_t k_joined = 4;
  constexpr std::size_t k_step_count = 5;
  auto make_step = []() -> decltype(auto) {
    return vkexec::make_pass_adaptor(vkexec::detail::make_callback_pass_step(
      [](vkexec::context &, VkCommandBuffer, vkexec::detail::pass_cleanup &) -> vkexec::status { return {}; }));
  };
  auto expression = vkexec::detail::pass_fragment_root{} | make_step()
                    | vkexec::when_all(make_step() | make_step(), make_step()) | make_step();
  auto chain = vkexec::detail::collect_pass_chain(std::move(expression), vkexec::scheduler_env{});
  REQUIRE(
    chain.step_predecessors
    == std::vector<std::vector<std::size_t>>{ {}, { k_root }, { k_left_first }, { k_root }, { k_left_last, k_right } });
  REQUIRE(chain.frontier == std::vector<std::size_t>{ k_joined });
  STATIC_REQUIRE(std::tuple_size_v<decltype(chain.steps)> == k_step_count);
}

TEST_CASE("nested typed branches keep queue affinity scoped", "[vkexec][pass]")
{
  constexpr std::size_t k_root = 0;
  constexpr std::size_t k_left = 1;
  constexpr std::size_t k_nested_first = 2;
  constexpr std::size_t k_nested_second = 3;
  constexpr std::size_t k_nested_join = 4;
  constexpr std::size_t k_right = 5;
  constexpr std::size_t k_joined = 6;
  char graphics_storage{};
  char compute_storage{};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  vkexec::queue_ref const graphics{ .queue = reinterpret_cast<VkQueue>(&graphics_storage), .family = k_default_family };
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  vkexec::queue_ref const compute{ .queue = reinterpret_cast<VkQueue>(&compute_storage), .family = k_other_family };
  auto make_step = []() -> decltype(auto) {
    return vkexec::make_pass_adaptor(vkexec::detail::make_callback_pass_step(
      [](vkexec::context &, VkCommandBuffer, vkexec::detail::pass_cleanup &) -> vkexec::status { return {}; }));
  };
  auto expression =
    vkexec::detail::pass_fragment_root{} | vkexec::on_queue(graphics) | make_step()
    | vkexec::when_all(
      vkexec::on_queue(compute) | make_step() | vkexec::when_all(make_step(), make_step()) | make_step(), make_step())
    | make_step();
  auto chain = vkexec::detail::collect_pass_chain(std::move(expression), vkexec::scheduler_env{});
  REQUIRE(chain.step_predecessors
          == std::vector<std::vector<std::size_t>>{ {},
            { k_root },
            { k_left },
            { k_left },
            { k_nested_first, k_nested_second },
            { k_root },
            { k_nested_join, k_right } });
  auto expect_queue = [&chain](std::size_t index, vkexec::queue_ref expected) -> void {
    REQUIRE(chain.step_queues.at(index).has_value());
    REQUIRE(vkexec::detail::same_queue(*chain.step_queues.at(index), expected));
  };
  expect_queue(k_root, graphics);
  for (std::size_t index{ k_left }; index <= k_nested_join; ++index) { expect_queue(index, compute); }
  expect_queue(k_right, graphics);
  expect_queue(k_joined, graphics);
}

TEST_CASE("typed branch lowering materializes a static graph with terminal sinks", "[vkexec][pass]")
{
  constexpr std::size_t k_root = 0;
  constexpr std::size_t k_left = 1;
  constexpr std::size_t k_right = 2;
  constexpr std::size_t k_step_count = 3;
  auto make_step = []() -> decltype(auto) {
    return vkexec::make_pass_adaptor(vkexec::detail::make_callback_pass_step(
      [](vkexec::context &, VkCommandBuffer, vkexec::detail::pass_cleanup &) -> vkexec::status { return {}; }));
  };
  auto expression = vkexec::schedule_sender{} | make_step() | vkexec::when_all(make_step(), make_step());
  auto chain = vkexec::detail::collect_pass_chain(std::move(expression), vkexec::scheduler_env{});
  auto graph = vkexec::detail::materialize_pass_chain(std::move(chain));
  STATIC_REQUIRE(vkexec::detail::is_pass_graph_sender_v<decltype(graph)>);
  STATIC_REQUIRE(std::tuple_size_v<decltype(graph.steps)> == k_step_count);
  REQUIRE(graph.step_predecessors == std::vector<std::vector<std::size_t>>{ {}, { k_root }, { k_root } });
  REQUIRE(graph.step_predecessors.at(k_left) == graph.step_predecessors.at(k_right));
}

TEST_CASE("resource tracking plans same queue image transitions", "[vkexec][resource-state][pass]")
{
  char storage{};
  vkexec::detail::resource_state_tracker tracker{ k_pair_step_count };
  REQUIRE(tracker.use(vkexec::write(fake_image(&storage), k_color_range, vkexec::image_usage::color_attachment),
    k_first_step,
    k_default_family));
  REQUIRE(tracker.use(vkexec::read(fake_image(&storage), k_color_range, vkexec::image_usage::sampled_fragment),
    k_second_step,
    k_default_family));

  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.dependencies.size() == k_single_step_count);
  REQUIRE(plan.dependencies.front().first == k_first_step);
  REQUIRE(plan.dependencies.front().second == k_second_step);
  REQUIRE(plan.steps.at(k_first_step).images_before.size() == 1);
  REQUIRE(plan.steps.at(k_first_step).images_before.front().old_layout == VK_IMAGE_LAYOUT_UNDEFINED);
  REQUIRE(plan.steps.at(k_second_step).images_before.size() == 1);
  auto const &barrier = plan.steps.at(k_second_step).images_before.front();
  REQUIRE(barrier.old_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  REQUIRE(barrier.new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  REQUIRE(barrier.src_access == VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  REQUIRE(barrier.dst_access == VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  REQUIRE(barrier.src_queue_family == VK_QUEUE_FAMILY_IGNORED);
}

TEST_CASE("buffer readers fan out and a writer joins their frontier", "[vkexec][resource-state][pass]")
{
  char storage{};
  char first_queue_storage{};
  char second_queue_storage{};
  auto *const buffer = fake_buffer(&storage);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  auto *const first_queue = reinterpret_cast<VkQueue>(&first_queue_storage);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  auto *const second_queue = reinterpret_cast<VkQueue>(&second_queue_storage);
  vkexec::detail::resource_state_tracker tracker{ k_dag_step_count };
  REQUIRE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::storage_compute), k_first_step, k_default_family));
  REQUIRE(tracker.use(
    vkexec::read(buffer, vkexec::buffer_usage::uniform_compute), k_second_step, k_default_family, first_queue));
  REQUIRE(tracker.use(
    vkexec::read(buffer, vkexec::buffer_usage::transfer_source), k_third_step, k_default_family, second_queue));
  REQUIRE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::storage_compute), k_fourth_step, k_default_family));

  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.dependencies
          == std::vector<std::pair<std::size_t, std::size_t>>{ { k_first_step, k_second_step },
            { k_first_step, k_third_step },
            { k_second_step, k_fourth_step },
            { k_third_step, k_fourth_step } });
  REQUIRE(plan.steps.at(k_fourth_step).buffers_before.size() == 1);
  auto const &barrier = plan.steps.at(k_fourth_step).buffers_before.front();
  REQUIRE(barrier.src_access == (VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT));
}

TEST_CASE("first buffer reads form a frontier", "[vkexec][resource-state][pass]")
{
  char storage{};
  char first_queue_storage{};
  char second_queue_storage{};
  auto *const buffer = fake_buffer(&storage);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  auto *const first_queue = reinterpret_cast<VkQueue>(&first_queue_storage);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  auto *const second_queue = reinterpret_cast<VkQueue>(&second_queue_storage);
  vkexec::detail::resource_state_tracker tracker{ k_dag_step_count };
  REQUIRE(tracker.use(
    vkexec::read(buffer, vkexec::buffer_usage::uniform_compute), k_first_step, k_default_family, first_queue));
  REQUIRE(tracker.use(
    vkexec::read(buffer, vkexec::buffer_usage::uniform_fragment), k_second_step, k_default_family, second_queue));
  REQUIRE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::storage_compute), k_third_step, k_default_family));

  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.dependencies
          == std::vector<std::pair<std::size_t, std::size_t>>{
            { k_first_step, k_third_step }, { k_second_step, k_third_step } });
}

TEST_CASE("imported image reads in an existing layout form a frontier", "[vkexec][resource-state][pass]")
{
  char storage{};
  char first_queue_storage{};
  char second_queue_storage{};
  auto *const image = fake_image(&storage);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  auto *const first_queue = reinterpret_cast<VkQueue>(&first_queue_storage);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  auto *const second_queue = reinterpret_cast<VkQueue>(&second_queue_storage);
  vkexec::detail::resource_state_tracker tracker{ k_dag_step_count };
  REQUIRE(tracker.use(
    vkexec::read(image, k_color_range, vkexec::image_usage::sampled_compute, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
    k_first_step,
    k_default_family,
    first_queue));
  REQUIRE(tracker.use(vkexec::read(image, k_color_range, vkexec::image_usage::sampled_fragment),
    k_second_step,
    k_default_family,
    second_queue));
  REQUIRE(tracker.use(
    vkexec::write(image, k_color_range, vkexec::image_usage::storage_compute), k_third_step, k_default_family));

  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.dependencies
          == std::vector<std::pair<std::size_t, std::size_t>>{
            { k_first_step, k_third_step }, { k_second_step, k_third_step } });
}

TEST_CASE("image layout change anchors a new read frontier", "[vkexec][resource-state][pass]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_dag_step_count };
  REQUIRE(tracker.use(
    vkexec::write(image, k_color_range, vkexec::image_usage::storage_compute), k_first_step, k_default_family));
  REQUIRE(tracker.use(
    vkexec::read(image, k_color_range, vkexec::image_usage::sampled_compute), k_second_step, k_default_family));
  REQUIRE(tracker.use(
    vkexec::read(image, k_color_range, vkexec::image_usage::sampled_fragment), k_third_step, k_default_family));
  REQUIRE(tracker.use(
    vkexec::write(image, k_color_range, vkexec::image_usage::storage_compute), k_fourth_step, k_default_family));

  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.dependencies
          == std::vector<std::pair<std::size_t, std::size_t>>{ { k_first_step, k_second_step },
            { k_second_step, k_third_step },
            { k_second_step, k_fourth_step },
            { k_third_step, k_fourth_step } });
  REQUIRE(plan.steps.at(k_fourth_step).images_before.size() == 1);
  REQUIRE(plan.steps.at(k_fourth_step).images_before.front().old_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}


// NOLINTEND(readability-function-cognitive-complexity)
