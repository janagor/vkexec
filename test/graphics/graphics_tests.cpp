// NOLINTBEGIN(readability-function-cognitive-complexity)

#include "test_helpers.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexec/execution.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/resource_state_tracker.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>
#include <vkexec_graphics/pass.hpp>
#include <vkexec_graphics/submit.hpp>
#include <vkexec_graphics/swapchain.hpp>

#include <vulkan/vulkan_core.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace {

constexpr float k_clear_r = 0.1F;
constexpr float k_clear_g = 0.2F;
constexpr float k_clear_b = 0.3F;
constexpr float k_clear_a = 0.4F;
constexpr std::size_t k_first_step = 0;
constexpr std::size_t k_second_step = 1;
constexpr std::size_t k_one_step_count = 1;
constexpr std::size_t k_two_step_count = 2;
constexpr std::uint32_t k_vertex_count = 1;
constexpr std::uint32_t k_one_binding = 1;
constexpr std::uint32_t k_no_colors = 0;
constexpr std::uint32_t k_graphics_family = 1;
constexpr std::uint32_t k_compute_family = 2;
constexpr std::uint32_t k_sampled_slot = 0;
constexpr VkImageSubresourceRange k_color_range{
  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
  .baseMipLevel = 0,
  .levelCount = 1,
  .baseArrayLayer = 0,
  .layerCount = 1,
};
constexpr VkImageSubresourceRange k_depth_range{
  .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
  .baseMipLevel = 0,
  .levelCount = 1,
  .baseArrayLayer = 0,
  .layerCount = 1,
};

[[nodiscard]] auto fake_image(void *storage) -> VkImage { return static_cast<VkImage>(storage); }
[[nodiscard]] auto fake_buffer(void *storage) -> VkBuffer { return static_cast<VkBuffer>(storage); }

}// namespace

TEST_CASE("graphics pass infers attachment and mesh uses", "[vkexec][graphics][pass]")
{
  char color_storage{};
  char vertex_storage{};
  char index_storage{};
  auto *const color = fake_image(&color_storage);
  auto *const vertex = fake_buffer(&vertex_storage);
  auto *const index = fake_buffer(&index_storage);
  vkexec::handles::graphics_pipeline const pipe{};
  auto bind = vkexec::bind_graphics(pipe);
  vkexec::graphics_target target{};
  target.colors.push_back(vkexec::graphics_attachment{ .image = color,
    .range = k_color_range,
    .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
    .render_pass_initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
    .final_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    .access = vkexec::resource_access::write });
  vkexec::detail::graphics_pass_step<vkexec::mesh_draw> const step{
    .target = std::move(target),
    .bind = std::move(bind),
    .draw = vkexec::mesh_draw{ .vertex_buffer = vertex, .index_buffer = index },
  };
  vkexec::detail::resource_state_tracker tracker{ k_two_step_count };
  REQUIRE(tracker.use(
    vkexec::write(color, k_color_range, vkexec::image_usage::storage_compute), k_first_step, k_graphics_family));
  REQUIRE(tracker.use(vkexec::write(vertex, vkexec::buffer_usage::storage_compute), k_first_step, k_graphics_family));
  REQUIRE(tracker.use(vkexec::write(index, vkexec::buffer_usage::storage_compute), k_first_step, k_graphics_family));
  REQUIRE(step.declare_resources(tracker, k_second_step, vkexec::queue_ref{ .family = k_graphics_family }));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_second_step).images_before.size() == k_one_step_count);
  REQUIRE(plan.steps.at(k_second_step).images_before.front().dst_access == VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  REQUIRE(plan.steps.at(k_second_step).buffers_before.size() == k_two_step_count);
  REQUIRE(
    plan.steps.at(k_second_step).buffers_before.at(k_first_step).dst_access == VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
  REQUIRE(plan.steps.at(k_second_step).buffers_before.at(k_second_step).dst_access == VK_ACCESS_2_INDEX_READ_BIT);
}

TEST_CASE("graphics pass infers depth read and write access", "[vkexec][graphics][pass]")
{
  char depth_storage{};
  auto *const depth = fake_image(&depth_storage);
  vkexec::handles::graphics_pipeline pipe{};
  pipe.cfg.color_attachment_count = k_no_colors;
  pipe.cfg.depth_test = true;
  pipe.cfg.depth_write = true;
  auto bind = vkexec::bind_graphics(pipe);
  vkexec::graphics_target target{};
  target.depth = vkexec::graphics_attachment{ .image = depth,
    .range = k_depth_range,
    .initial_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    .render_pass_initial_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    .final_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    .access = vkexec::resource_access::read_write };
  vkexec::detail::graphics_pass_step<std::uint32_t> const step{
    .target = std::move(target),
    .bind = std::move(bind),
    .draw = k_vertex_count,
  };
  vkexec::detail::resource_state_tracker tracker{ k_one_step_count };
  REQUIRE(step.declare_resources(tracker, k_first_step, vkexec::queue_ref{ .family = k_graphics_family }));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_first_step).images_before.front().dst_access
          == (VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT));
}

TEST_CASE("depth-tested graphics pass requires depth metadata", "[vkexec][graphics][pass]")
{
  vkexec::handles::graphics_pipeline pipe{};
  pipe.cfg.color_attachment_count = k_no_colors;
  pipe.cfg.depth_test = true;
  auto bind = vkexec::bind_graphics(pipe);
  vkexec::detail::graphics_pass_step<std::uint32_t> const step{
    .target = vkexec::graphics_target{},
    .bind = std::move(bind),
    .draw = k_vertex_count,
  };
  REQUIRE_FALSE(step.has_resource_declarations());
  vkexec::detail::resource_state_tracker tracker{ k_one_step_count };
  REQUIRE_FALSE(step.declare_resources(tracker, k_first_step, vkexec::queue_ref{ .family = k_graphics_family }));
}

TEST_CASE("graphics descriptors synchronize after compute writes", "[vkexec][graphics][pass]")
{
  char image_storage{};
  auto *const image = fake_image(&image_storage);
  vkexec::handles::graphics_pipeline pipe{};
  pipe.cfg.color_attachment_count = k_no_colors;
  pipe.binding_count = k_one_binding;
  pipe.binding_slots = { k_sampled_slot };
  pipe.binding_kinds = { vkexec::resource_kind::sampled_image };
  auto const table = vkexec::bindings(vkexec::resource_binding{ .slot = k_sampled_slot,
    .resource =
      vkexec::sampled_image_resource(image, VK_NULL_HANDLE, k_color_range, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) });
  auto bind = vkexec::bind_graphics(pipe, VK_NULL_HANDLE, table);
  vkexec::detail::graphics_pass_step<std::uint32_t> const step{
    .target = vkexec::graphics_target{},
    .bind = std::move(bind),
    .draw = k_vertex_count,
  };
  vkexec::detail::resource_state_tracker tracker{ k_two_step_count };
  REQUIRE(tracker.use(
    vkexec::write(image, k_color_range, vkexec::image_usage::storage_compute), k_first_step, k_compute_family));
  REQUIRE(step.declare_resources(tracker, k_second_step, vkexec::queue_ref{ .family = k_graphics_family }));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_first_step).images_after.size() == k_one_step_count);
  REQUIRE(plan.steps.at(k_second_step).images_before.front().dst_access == VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

TEST_CASE("opaque graphics descriptors require explicit complete resources", "[vkexec][graphics][pass]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::handles::graphics_pipeline pipe{};
  pipe.binding_count = k_one_binding;
  pipe.cfg.color_attachment_count = k_no_colors;
  auto const raw = vkexec::bind_graphics(pipe);
  vkexec::detail::graphics_pass_step<std::uint32_t> const missing{ .target = {}, .bind = raw, .draw = k_vertex_count };
  vkexec::detail::resource_state_tracker tracker{ k_two_step_count };
  REQUIRE_FALSE(vkexec::detail::declare_step_resources(
    missing, tracker, k_second_step, vkexec::queue_ref{ .family = k_graphics_family }, false));
  REQUIRE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::storage_compute), k_first_step, k_graphics_family));
  vkexec::detail::graphics_pass_step<std::uint32_t> const declared{ .target = {},
    .bind = raw.with_resources({}, { vkexec::read(buffer, vkexec::buffer_usage::storage_graphics) }),
    .draw = k_vertex_count };
  REQUIRE(vkexec::detail::declare_step_resources(
    declared, tracker, k_second_step, vkexec::queue_ref{ .family = k_graphics_family }, false));
  auto const plan = std::move(tracker).finish();
  REQUIRE_FALSE(plan.steps.at(k_second_step).buffers_before.empty());
}

TEST_CASE("graphics pass rejects incomplete descriptor metadata", "[vkexec][graphics][pass]")
{
  char color_storage{};
  vkexec::handles::graphics_pipeline pipe{};
  pipe.binding_count = k_one_binding;
  auto bind = vkexec::bind_graphics(pipe);
  vkexec::graphics_target target{};
  target.colors.push_back(vkexec::graphics_attachment{ .image = fake_image(&color_storage),
    .range = k_color_range,
    .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
    .render_pass_initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
    .final_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    .access = vkexec::resource_access::write });
  vkexec::detail::graphics_pass_step<std::uint32_t> const step{
    .target = std::move(target),
    .bind = std::move(bind),
    .draw = k_vertex_count,
  };
  REQUIRE_FALSE(step.has_resource_declarations());
  vkexec::detail::resource_state_tracker tracker{ k_one_step_count };
  REQUIRE_FALSE(vkexec::detail::declare_step_resources(
    step, tracker, k_first_step, vkexec::queue_ref{ .family = k_graphics_family }, true));
  REQUIRE_FALSE(step.declare_resources(tracker, k_first_step, vkexec::queue_ref{ .family = k_graphics_family }));
}

TEST_CASE("render-pass final layout becomes the next graph layout", "[vkexec][graphics][pass]")
{
  char image_storage{};
  auto *const image = fake_image(&image_storage);
  vkexec::handles::graphics_pipeline const pipe{};
  auto bind = vkexec::bind_graphics(pipe);
  vkexec::graphics_target target{};
  target.colors.push_back(vkexec::graphics_attachment{ .image = image,
    .range = k_color_range,
    .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
    .render_pass_initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
    .final_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    .access = vkexec::resource_access::write });
  vkexec::detail::graphics_pass_step<std::uint32_t> const step{
    .target = std::move(target),
    .bind = std::move(bind),
    .draw = k_vertex_count,
  };
  vkexec::detail::resource_state_tracker tracker{ k_two_step_count };
  REQUIRE(step.declare_resources(tracker, k_first_step, vkexec::queue_ref{ .family = k_graphics_family }));
  REQUIRE(tracker.use(
    vkexec::read(image, k_color_range, vkexec::image_usage::sampled_fragment), k_second_step, k_graphics_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_second_step).images_before.front().old_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

TEST_CASE("graphics pass enters the render pass initial layout", "[vkexec][graphics][pass]")
{
  char image_storage{};
  auto *const image = fake_image(&image_storage);
  vkexec::handles::graphics_pipeline const pipe{};
  auto bind = vkexec::bind_graphics(pipe);
  vkexec::graphics_target target{};
  target.colors.push_back(vkexec::graphics_attachment{ .image = image,
    .range = k_color_range,
    .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
    .render_pass_initial_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    .final_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    .access = vkexec::resource_access::write });
  vkexec::detail::graphics_pass_step<std::uint32_t> const step{
    .target = std::move(target),
    .bind = std::move(bind),
    .draw = k_vertex_count,
  };
  vkexec::detail::resource_state_tracker tracker{ k_two_step_count };
  REQUIRE(tracker.use(
    vkexec::write(image, k_color_range, vkexec::image_usage::storage_compute), k_first_step, k_graphics_family));
  REQUIRE(step.declare_resources(tracker, k_second_step, vkexec::queue_ref{ .family = k_graphics_family }));
  auto const plan = std::move(tracker).finish();
  auto const &barrier = plan.steps.at(k_second_step).images_before.front();
  REQUIRE(barrier.old_layout == VK_IMAGE_LAYOUT_GENERAL);
  REQUIRE(barrier.new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

TEST_CASE("undefined render-pass initial layout leaves its transition implicit", "[vkexec][graphics][pass]")
{
  char image_storage{};
  auto *const image = fake_image(&image_storage);
  vkexec::handles::graphics_pipeline const pipe{};
  auto bind = vkexec::bind_graphics(pipe);
  vkexec::graphics_target target{};
  target.colors.push_back(vkexec::graphics_attachment{ .image = image,
    .range = k_color_range,
    .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
    .render_pass_initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
    .final_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    .access = vkexec::resource_access::write });
  vkexec::detail::graphics_pass_step<std::uint32_t> const step{
    .target = std::move(target),
    .bind = std::move(bind),
    .draw = k_vertex_count,
  };
  vkexec::detail::resource_state_tracker tracker{ k_two_step_count };
  REQUIRE(tracker.use(
    vkexec::write(image, k_color_range, vkexec::image_usage::storage_compute), k_first_step, k_graphics_family));
  REQUIRE(step.declare_resources(tracker, k_second_step, vkexec::queue_ref{ .family = k_graphics_family }));
  auto const plan = std::move(tracker).finish();
  auto const &barrier = plan.steps.at(k_second_step).images_before.front();
  REQUIRE(barrier.old_layout == VK_IMAGE_LAYOUT_GENERAL);
  REQUIRE(barrier.new_layout == VK_IMAGE_LAYOUT_GENERAL);
  REQUIRE(barrier.dst_access == VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
}

TEST_CASE("first attachment use with undefined render-pass layout has no graph transition", "[vkexec][graphics][pass]")
{
  char image_storage{};
  vkexec::handles::graphics_pipeline const pipe{};
  auto bind = vkexec::bind_graphics(pipe);
  vkexec::graphics_target target{};
  target.colors.push_back(vkexec::graphics_attachment{ .image = fake_image(&image_storage),
    .range = k_color_range,
    .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
    .render_pass_initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
    .final_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    .access = vkexec::resource_access::write });
  vkexec::detail::graphics_pass_step<std::uint32_t> const step{
    .target = std::move(target),
    .bind = std::move(bind),
    .draw = k_vertex_count,
  };
  vkexec::detail::resource_state_tracker tracker{ k_one_step_count };
  REQUIRE(step.declare_resources(tracker, k_first_step, vkexec::queue_ref{ .family = k_graphics_family }));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_first_step).images_before.empty());
}

TEST_CASE("graphics pass requires explicit render-pass attachment layouts", "[vkexec][graphics][pass]")
{
  char image_storage{};
  vkexec::handles::graphics_pipeline const pipe{};
  auto bind = vkexec::bind_graphics(pipe);
  vkexec::graphics_target target{};
  target.colors.push_back(vkexec::graphics_attachment{ .image = fake_image(&image_storage),
    .range = k_color_range,
    .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
    .render_pass_initial_layout = std::nullopt,
    .final_layout = VK_IMAGE_LAYOUT_UNDEFINED,
    .access = vkexec::resource_access::write });
  vkexec::detail::graphics_pass_step<std::uint32_t> const step{
    .target = std::move(target),
    .bind = std::move(bind),
    .draw = k_vertex_count,
  };
  REQUIRE_FALSE(step.has_resource_declarations());
  vkexec::detail::resource_state_tracker tracker{ k_one_step_count };
  REQUIRE_FALSE(step.declare_resources(tracker, k_first_step, vkexec::queue_ref{ .family = k_graphics_family }));
}

TEST_CASE("present_options carries an extension-neutral pNext chain", "[vkexec][graphics]")
{
  VkPresentIdKHR present_id{};
  present_id.sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR;
  vkexec::present_options const options{ .p_next = &present_id };

  REQUIRE(options.p_next == &present_id);
}

TEST_CASE("swapchain create info carries Vulkan create flags", "[vkexec][graphics]")
{
  vkexec::swapchain_create_info const defaults{};
  REQUIRE(defaults.flags == 0);

  vkexec::swapchain_create_info configured{};
  configured.flags = VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR;
  REQUIRE(configured.flags == VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR);
}

// NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
TEST_CASE("make_clear_values maps pipeline config to Vulkan clears", "[vkexec][graphics]")
{
  vkexec::graphics_pipeline_config cfg{};
  cfg.clear_r = k_clear_r;
  cfg.clear_g = k_clear_g;
  cfg.clear_b = k_clear_b;
  cfg.clear_a = k_clear_a;

  auto const clears = vkexec::make_clear_values(cfg);

  REQUIRE(clears.at(0).color.float32[0] == Catch::Approx(k_clear_r));
  REQUIRE(clears.at(0).color.float32[1] == Catch::Approx(k_clear_g));
  REQUIRE(clears.at(0).color.float32[2] == Catch::Approx(k_clear_b));
  REQUIRE(clears.at(0).color.float32[3] == Catch::Approx(k_clear_a));
  REQUIRE(clears.at(1).depthStencil.depth == Catch::Approx(vkexec::k_depth_clear_value));
  REQUIRE(clears.at(1).depthStencil.stencil == vkexec::k_stencil_clear_value);
}

TEST_CASE("make_clear_values uses default clear color from config", "[vkexec][graphics]")
{
  vkexec::graphics_pipeline_config const cfg{};
  auto const clears = vkexec::make_clear_values(cfg);

  REQUIRE(clears.at(0).color.float32[0] == Catch::Approx(vkexec::k_default_clear_r));
  REQUIRE(clears.at(0).color.float32[1] == Catch::Approx(vkexec::k_default_clear_g));
  REQUIRE(clears.at(0).color.float32[2] == Catch::Approx(vkexec::k_default_clear_b));
  REQUIRE(clears.at(0).color.float32[3] == Catch::Approx(vkexec::k_default_clear_a));
}
// NOLINTEND(cppcoreguidelines-pro-type-union-access)

TEST_CASE("draw | submit yields stop-aware async sender", "[vkexec][graphics][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  vkexec::draw_sender const sync{
    .state = {},
    .win = nullptr,
    .pipeline = nullptr,
    .vertex_count = 0,
  };
  auto const async_sender = sync | vkexec::submit;

  STATIC_REQUIRE(std::same_as<std::remove_cvref_t<decltype(async_sender)>, vkexec::draw_async_sender>);

  auto const completion = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(async_sender));
  REQUIRE(completion == sched);

  using signatures = vkexec::draw_async_sender::completion_signatures;
  STATIC_REQUIRE(std::same_as<signatures,
    stdexec::
      completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(vkexec::error), stdexec::set_stopped_t()>>);
}

TEST_CASE("draw senders retain runtime after context destruction", "[vkexec][graphics][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  std::weak_ptr<vkexec::detail::context_state> const runtime = vkexec::detail::context_access::state(*ctx);
  auto sender = stdexec::schedule(ctx->get_scheduler()) | vkexec::draw_closure{};
  auto async_sender = std::move(sender) | vkexec::submit;
  ctx.reset();

  REQUIRE(!runtime.expired());
  auto const completion = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(async_sender));
  REQUIRE(vkexec::detail::scheduler_access::state(completion).get() == runtime.lock().get());
}

TEST_CASE("graphics attachment final layout leaves other mips unchanged", "[vkexec][graphics][pass]")
{
  constexpr std::uint32_t k_two_mips = 2;
  constexpr std::uint32_t k_attachment_mip = 1;
  constexpr std::size_t k_third_step = 2;
  constexpr std::size_t k_three_steps = 3;
  char storage{};
  auto *const image = fake_image(&storage);
  auto whole = k_color_range;
  whole.levelCount = k_two_mips;
  auto attachment_range = k_color_range;
  attachment_range.baseMipLevel = k_attachment_mip;
  vkexec::handles::graphics_pipeline const pipe{};
  vkexec::graphics_target target{};
  target.colors.push_back(vkexec::graphics_attachment{ .image = image,
    .range = attachment_range,
    .render_pass_initial_layout = VK_IMAGE_LAYOUT_UNDEFINED,
    .final_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    .access = vkexec::resource_access::write });
  vkexec::detail::graphics_pass_step<std::uint32_t> const step{
    .target = std::move(target), .bind = vkexec::bind_graphics(pipe), .draw = k_vertex_count
  };
  vkexec::detail::resource_state_tracker tracker{ k_three_steps };
  REQUIRE(
    tracker.use(vkexec::write(image, whole, vkexec::image_usage::storage_compute), k_first_step, k_graphics_family));
  REQUIRE(step.declare_resources(tracker, k_second_step, vkexec::queue_ref{ .family = k_graphics_family }));
  REQUIRE(
    tracker.use(vkexec::read(image, whole, vkexec::image_usage::sampled_fragment), k_third_step, k_graphics_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_second_step).images_before.front().range.baseMipLevel == k_attachment_mip);
  REQUIRE(plan.steps.at(k_third_step).images_before.size() == k_two_mips);
  for (auto const &barrier : plan.steps.at(k_third_step).images_before) {
    REQUIRE(barrier.old_layout
            == (barrier.range.baseMipLevel == k_attachment_mip ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                                               : VK_IMAGE_LAYOUT_GENERAL));
  }
}

// NOLINTEND(readability-function-cognitive-complexity)
