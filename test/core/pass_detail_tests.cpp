#include <catch2/catch_test_macros.hpp>

#include <vkexec/detail/resource_state_tracker.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/result.hpp>
#include <vkexec/submit_scope.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace {

constexpr VkDeviceSize k_byte_size = 64;
constexpr std::size_t k_first_step = 0;
constexpr std::size_t k_second_step = 1;
constexpr std::size_t k_pair_step_count = 2;
constexpr std::size_t k_single_step_count = 1;
constexpr std::size_t k_dag_step_count = 4;
constexpr std::uint32_t k_default_family = 0;
constexpr std::uint32_t k_sampled_binding = 0;

[[nodiscard]] auto fake_buffer(void *storage) -> VkBuffer { return static_cast<VkBuffer>(storage); }

[[nodiscard]] auto fake_image(void *storage) -> VkImage { return static_cast<VkImage>(storage); }

constexpr VkImageSubresourceRange k_color_range{
  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
  .baseMipLevel = 0,
  .levelCount = 1,
  .baseArrayLayer = 0,
  .layerCount = 1,
};

}// namespace

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

TEST_CASE("pass batches group adjacent steps by resolved queue", "[vkexec][pass]")
{
  char graphics_storage{};
  char compute_storage{};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  vkexec::queue_ref const graphics{ .queue = reinterpret_cast<VkQueue>(&graphics_storage), .family = 0 };
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  vkexec::queue_ref const compute{ .queue = reinterpret_cast<VkQueue>(&compute_storage), .family = 1 };
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

TEST_CASE("DAG planner keeps independent branches separate", "[vkexec][pass]")
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

TEST_CASE("resource tracking plans same queue image transitions", "[vkexec][pass]")
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

TEST_CASE("compute binding preserves a sampled descriptor's GENERAL layout", "[vkexec][pass]")
{
  char image_storage{};
  auto *const image = fake_image(&image_storage);
  vkexec::handles::compute_pipeline pipe{};
  pipe.binding_accesses = { vkexec::buffer_access::readonly };
  pipe.binding_kinds = { vkexec::resource_kind::sampled_image };
  auto const table = vkexec::bindings(vkexec::resource_binding{ .slot = k_sampled_binding,
    .resource = vkexec::sampled_image_resource(
      image, VK_NULL_HANDLE, k_color_range, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) });
  auto const bind = vkexec::bind_compute(pipe, VK_NULL_HANDLE, table);
  REQUIRE(bind.complete_resource_metadata);
  REQUIRE(bind.images.size() == k_single_step_count);

  vkexec::detail::resource_state_tracker tracker{ k_single_step_count };
  REQUIRE(tracker.use(bind.images.front(), k_first_step, k_default_family));
  auto const plan = std::move(tracker).finish();
  auto const &barrier = plan.steps.front().images_before.front();
  REQUIRE(barrier.old_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  REQUIRE(barrier.new_layout == VK_IMAGE_LAYOUT_GENERAL);
  REQUIRE(barrier.dst_access == VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

TEST_CASE("compute binding rejects missing storage buffer metadata", "[vkexec][pass]")
{
  constexpr std::uint32_t k_storage_binding = 0;
  vkexec::handles::compute_pipeline pipe{};
  pipe.binding_accesses = { vkexec::buffer_access::readonly };
  auto const table = vkexec::bindings(vkexec::resource_binding{
    .slot = k_storage_binding, .resource = vkexec::buffer_resource(VK_NULL_HANDLE, sizeof(std::uint32_t)) });
  auto const bind = vkexec::bind_compute(pipe, VK_NULL_HANDLE, table);
  REQUIRE(bind.resource_metadata);
  REQUIRE_FALSE(bind.complete_resource_metadata);
}

TEST_CASE("resource tracking plans cross family release and acquire", "[vkexec][pass]")
{
  char storage{};
  constexpr std::uint32_t k_graphics_family = 2;
  constexpr std::uint32_t k_compute_family = 3;
  vkexec::detail::resource_state_tracker tracker{ k_pair_step_count };
  REQUIRE(tracker.use(vkexec::write(fake_image(&storage), k_color_range, vkexec::image_usage::color_attachment),
    k_first_step,
    k_graphics_family));
  REQUIRE(tracker.use(vkexec::read(fake_image(&storage), k_color_range, vkexec::image_usage::sampled_compute),
    k_second_step,
    k_compute_family));

  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_first_step).images_after.size() == 1);
  REQUIRE(plan.steps.at(k_second_step).images_before.size() == 1);
  auto const &release = plan.steps.at(k_first_step).images_after.front();
  auto const &acquire = plan.steps.at(k_second_step).images_before.front();
  REQUIRE(release.src_queue_family == k_graphics_family);
  REQUIRE(release.dst_queue_family == k_compute_family);
  REQUIRE(release.dst_stage == VK_PIPELINE_STAGE_2_NONE);
  REQUIRE(acquire.src_stage == VK_PIPELINE_STAGE_2_NONE);
  REQUIRE(acquire.dst_stage == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  REQUIRE(release.new_layout == acquire.new_layout);
}

TEST_CASE("resource tracking requires an initial layout for first reads", "[vkexec][pass]")
{
  char storage{};
  vkexec::detail::resource_state_tracker tracker{ k_single_step_count };
  REQUIRE_FALSE(tracker.use(vkexec::read(fake_image(&storage), k_color_range, vkexec::image_usage::sampled_compute),
    k_first_step,
    k_default_family));
  REQUIRE(tracker.use(vkexec::read(fake_image(&storage),
                        k_color_range,
                        vkexec::image_usage::sampled_compute,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
    k_first_step,
    k_default_family));
}

TEST_CASE("resource tracking uses semaphore ordering across queues in one family", "[vkexec][pass]")
{
  char image_storage{};
  char first_queue_storage{};
  char second_queue_storage{};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  auto *first_queue = reinterpret_cast<VkQueue>(&first_queue_storage);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  auto *second_queue = reinterpret_cast<VkQueue>(&second_queue_storage);
  vkexec::detail::resource_state_tracker tracker{ k_pair_step_count };
  REQUIRE(tracker.use(vkexec::write(fake_image(&image_storage), k_color_range, vkexec::image_usage::color_attachment),
    k_first_step,
    k_default_family,
    first_queue));
  REQUIRE(tracker.use(vkexec::read(fake_image(&image_storage), k_color_range, vkexec::image_usage::sampled_compute),
    k_second_step,
    k_default_family,
    second_queue));

  auto const plan = std::move(tracker).finish();
  auto const &barrier = plan.steps.at(k_second_step).images_before.front();
  REQUIRE(barrier.src_stage == VK_PIPELINE_STAGE_2_NONE);
  REQUIRE(barrier.src_access == VK_ACCESS_2_NONE);
  REQUIRE(barrier.dst_stage == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  REQUIRE(barrier.src_queue_family == VK_QUEUE_FAMILY_IGNORED);
}

TEST_CASE("resource tracking rejects access incompatible with image usage", "[vkexec][pass]")
{
  char storage{};
  vkexec::detail::resource_state_tracker tracker{ k_single_step_count };
  auto *const image = fake_image(&storage);
  REQUIRE_FALSE(tracker.use(
    vkexec::write(image, k_color_range, vkexec::image_usage::sampled_fragment), k_first_step, k_default_family));
  REQUIRE_FALSE(tracker.use(
    vkexec::read(image, k_color_range, vkexec::image_usage::transfer_destination), k_first_step, k_default_family));
}

TEST_CASE("resource tracking rejects access incompatible with buffer usage", "[vkexec][pass]")
{
  char storage{};
  vkexec::detail::resource_state_tracker tracker{ k_single_step_count };
  auto *const buffer = fake_buffer(&storage);
  REQUIRE_FALSE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::vertex), k_first_step, k_default_family));
  REQUIRE_FALSE(
    tracker.use(vkexec::read(buffer, vkexec::buffer_usage::transfer_destination), k_first_step, k_default_family));
}

TEST_CASE("uniform buffer usage selects one shader stage", "[vkexec][pass]")
{
  char compute_storage{};
  char vertex_storage{};
  auto const compute = vkexec::read(fake_buffer(&compute_storage), vkexec::buffer_usage::uniform_compute);
  auto const vertex = vkexec::read(fake_buffer(&vertex_storage), vkexec::buffer_usage::uniform_vertex);
  REQUIRE(vkexec::detail::buffer_scope(compute).stage == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  REQUIRE(vkexec::detail::buffer_scope(vertex).stage == VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
}

TEST_CASE("resource tracking rejects two declarations in one pass", "[vkexec][pass]")
{
  char image_storage{};
  char buffer_storage{};
  auto *const image = fake_image(&image_storage);
  auto *const buffer = fake_buffer(&buffer_storage);
  vkexec::detail::resource_state_tracker tracker{ k_single_step_count };
  REQUIRE(tracker.use(
    vkexec::write(image, k_color_range, vkexec::image_usage::color_attachment), k_first_step, k_default_family));
  REQUIRE_FALSE(tracker.use(
    vkexec::read(image, k_color_range, vkexec::image_usage::sampled_fragment), k_first_step, k_default_family));
  REQUIRE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::storage_compute), k_first_step, k_default_family));
  REQUIRE_FALSE(
    tracker.use(vkexec::read(buffer, vkexec::buffer_usage::storage_compute), k_first_step, k_default_family));
}
