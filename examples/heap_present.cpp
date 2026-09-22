#include "sync_wait_helpers.hpp"
#include <vkexec/barrier.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/gpu_buffer.hpp>
#include <vkexec/image.hpp>
#include <vkexec/image_view.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>
#include <vkexec/submit.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_extensions/descriptor_heap/buffer.hpp>
#include <vkexec_extensions/descriptor_heap/compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/descriptor_heap.hpp>
#include <vkexec_extensions/descriptor_heap/extension.hpp>
#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/heap_graphics_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/pass.hpp>
#include <vkexec_extensions/descriptor_heap/strategy.hpp>
#include <vkexec_extensions/dynamic_rendering/rendering.hpp>
#include <vkexec_extensions/extension.hpp>
#include <vkexec_features/bundles/vulkan_13.hpp>

#include <stdexec/execution.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/presenter.hpp>
#include <vkexec_graphics/triangle_shaders.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <string_view>
#include <utility>

namespace ex = stdexec;
namespace {

constexpr std::uint32_t k_width = 64;
constexpr std::uint32_t k_height = 64;
constexpr std::uint32_t k_present_frames = 3;
constexpr std::uint32_t k_triangle_vertices = 3;
constexpr std::uint32_t k_work_count = 64;
constexpr std::size_t k_heap_slots = 1;
constexpr VkDeviceSize k_storage_bytes = 256;
constexpr float k_clear_r = 0.15F;
constexpr float k_clear_g = 0.35F;
constexpr float k_clear_b = 0.55F;
constexpr float k_clear_a = 1.0F;
constexpr VkFormat k_heap_graphics_format = VK_FORMAT_R8G8B8A8_UNORM;
constexpr std::string_view k_heap_glsl = R"(#version 460
layout(local_size_x = 64) in;
void main() {}
)";

struct heap_push
{
  std::uint32_t count;
};

auto make_requirements() -> vkexec::vulkan_requirements
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 3;
  vkexec::feat::configure_vulkan_13(requirements);
  vkexec::ext::configure<vkexec::ext::descriptor_heap>(requirements);
  return requirements;
}

auto run_dynamic_rendering(vkexec::context &ctx) -> vkexec::status
{
  auto img = vkexec::examples::sync_wait_value(vkexec::factory::image(ctx,
    vkexec::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::image_usage::color_storage,
    }));
  auto view = vkexec::examples::sync_wait_value(vkexec::factory::image_view(ctx, img));

  auto cmd_result = ctx.allocate_command_buffer();
  if (!cmd_result) { return vkexec::fail(std::move(cmd_result.error())); }
  auto *cmd = vkexec::expected_take(cmd_result);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
    return vkexec::fail(VK_ERROR_UNKNOWN, "vkBeginCommandBuffer failed (dynamic rendering)");
  }

  vkexec::image_barrier(cmd,
    {
      .image = img.handle(),
      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
      .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      .dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .src_access = 0,
      .dst_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    });

  vkexec::color_attachment color{};
  color.view = view.handle();
  color.clear.color = { { k_clear_r, k_clear_g, k_clear_b, k_clear_a } };
  std::array<vkexec::color_attachment, 1> const colors{ color };
  if (auto began = vkexec::cmd_begin_rendering(cmd,
        vkexec::rendering_info{
          .extent = img.extent(),
          .color = colors,
        });
    !began) {
    ctx.free_command_buffer(cmd);
    return vkexec::fail(std::move(began.error()));
  }
  if (auto ended = vkexec::cmd_end_rendering(cmd); !ended) {
    ctx.free_command_buffer(cmd);
    return vkexec::fail(std::move(ended.error()));
  }

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    ctx.free_command_buffer(cmd);
    return vkexec::fail(VK_ERROR_UNKNOWN, "vkEndCommandBuffer failed");
  }
  std::array<VkCommandBuffer, 1> const cmds{ cmd };

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence{ VK_NULL_HANDLE };
  if (vkCreateFence(ctx.device(), &fence_info, nullptr, &fence) != VK_SUCCESS) {
    ctx.free_command_buffer(cmd);
    return vkexec::fail(VK_ERROR_UNKNOWN, "vkCreateFence failed (dynamic rendering)");
  }

  if (auto submitted = ctx.submit(vkexec::queue_submit{
        .command_buffers = cmds,
        .fence = fence,
        .queue = ctx.graphics_queue() != VK_NULL_HANDLE ? ctx.graphics_queue() : ctx.compute_queue(),
      });
    !submitted) {
    vkDestroyFence(ctx.device(), fence, nullptr);
    ctx.free_command_buffer(cmd);
    return vkexec::fail(std::move(submitted.error()));
  }
  if (vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
    vkDestroyFence(ctx.device(), fence, nullptr);
    ctx.free_command_buffer(cmd);
    return vkexec::fail(VK_ERROR_UNKNOWN, "vkWaitForFences failed (dynamic rendering)");
  }
  vkDestroyFence(ctx.device(), fence, nullptr);
  ctx.free_command_buffer(cmd);
  return {};
}

auto run_heap_graphics(vkexec::context &ctx) -> bool
{
  if (!vkexec::ext::available<vkexec::ext::descriptor_heap>(ctx)) { return false; }

  auto pipe = vkexec::examples::sync_wait_value(vkexec::factory::graphics_pipeline(vkexec::descriptor_heap,
    ctx,
    vkexec::shaders::k_triangle_vert,
    vkexec::shaders::k_triangle_frag,
    vkexec::heap_graphics_layout_desc{
      .blend = vkexec::blend_mode::premultiplied,
      .depth_test = false,
      .depth_write = false,
      .color_formats = { k_heap_graphics_format },
    }));

  auto img = vkexec::examples::sync_wait_value(vkexec::factory::image(ctx,
    vkexec::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::image_usage::color_storage,
      .format = k_heap_graphics_format,
    }));
  auto view = vkexec::examples::sync_wait_value(vkexec::factory::image_view(ctx, img));

  auto cmd_result = ctx.allocate_command_buffer();
  if (!cmd_result) { return false; }
  auto *cmd = vkexec::expected_take(cmd_result);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
    ctx.free_command_buffer(cmd);
    return false;
  }

  vkexec::image_barrier(cmd,
    {
      .image = img.handle(),
      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
      .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      .dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .src_access = 0,
      .dst_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    });

  vkexec::color_attachment color{};
  color.view = view.handle();
  color.clear.color = { { k_clear_r, k_clear_g, k_clear_b, k_clear_a } };
  std::array<vkexec::color_attachment, 1> const colors{ color };
  if (auto began = vkexec::cmd_begin_rendering(cmd,
        vkexec::rendering_info{
          .extent = img.extent(),
          .color = colors,
        });
    !began) {
    ctx.free_command_buffer(cmd);
    return false;
  }

  vkexec::record_draw(ctx, cmd, pipe.bind(), img.extent(), k_triangle_vertices);

  if (auto ended = vkexec::cmd_end_rendering(cmd); !ended) {
    ctx.free_command_buffer(cmd);
    return false;
  }

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    ctx.free_command_buffer(cmd);
    return false;
  }
  std::array<VkCommandBuffer, 1> const cmds{ cmd };

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence{ VK_NULL_HANDLE };
  if (vkCreateFence(ctx.device(), &fence_info, nullptr, &fence) != VK_SUCCESS) {
    ctx.free_command_buffer(cmd);
    return false;
  }

  if (auto submitted = ctx.submit(vkexec::queue_submit{
        .command_buffers = cmds,
        .fence = fence,
        .queue = ctx.graphics_queue() != VK_NULL_HANDLE ? ctx.graphics_queue() : ctx.compute_queue(),
      });
    !submitted) {
    vkDestroyFence(ctx.device(), fence, nullptr);
    ctx.free_command_buffer(cmd);
    return false;
  }
  if (vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
    vkDestroyFence(ctx.device(), fence, nullptr);
    ctx.free_command_buffer(cmd);
    return false;
  }
  vkDestroyFence(ctx.device(), fence, nullptr);
  ctx.free_command_buffer(cmd);
  return true;
}

auto run_heap_compute(vkexec::context &ctx) -> bool
{
  if (!vkexec::ext::available<vkexec::ext::descriptor_heap>(ctx)) { return false; }

  auto layout_result = vkexec::query_descriptor_heap_layout(ctx);
  if (!layout_result) { return false; }
  auto const &layout = vkexec::expected_get(layout_result);

  auto storage = vkexec::examples::sync_wait_value(vkexec::factory::gpu_buffer(ctx,
    vkexec::gpu_buffer_create_info{
      .size = k_storage_bytes,
      .memory = vkexec::gpu_buffer_memory::device_local,
      .shader_device_address = true,
    }));
  auto heap = vkexec::examples::sync_wait_value(
    vkexec::factory::descriptor_heap_buffer(ctx, vkexec::descriptor_heap_byte_size(layout, k_heap_slots)));

  auto const storage_addr = storage.device_address();
  if (!storage_addr) { return false; }
  auto mapped = heap.mapped();
  if (auto const written = vkexec::write_storage_buffer_descriptor(
        ctx, *storage_addr, storage.size(), mapped.subspan(0, layout.buffer_descriptor_size));
    !written) {
    return false;
  }

  auto pipe = vkexec::examples::sync_wait_value(vkexec::factory::compute_pipeline(vkexec::descriptor_heap,
    ctx,
    k_heap_glsl,
    vkexec::heap_layout_desc{ .specialization = {}, .local_size = vkexec::k_default_local_size },
    "heap_present.comp"));

  heap_push const params{ .count = k_work_count };
  auto outcome = vkexec::try_sync_wait(
    ex::schedule(ctx.get_scheduler()) | vkexec::compute_pass(vkexec::descriptor_heap, pipe, params, k_work_count));
  return !outcome.failed() && outcome.values.has_value() && !outcome.stopped;
}

auto present_frames(vkexec::presenter &win, vkexec::graphics_pipeline &pipeline) -> void
{
  for (std::uint32_t frame = 0; frame < k_present_frames; ++frame) {
    vkexec::examples::sync_wait_graph(
      ex::schedule(win.ctx().get_scheduler()) | vkexec::draw(win, pipeline, k_triangle_vertices) | vkexec::submit);
  }
}

}// namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
static auto run() -> int
{
  auto win = vkexec::examples::sync_wait_value(vkexec::factory::headless_presenter(vkexec::presenter::config{
    .width = k_width,
    .height = k_height,
    .validation_layers = false,
    .surface_instance_extensions = {},
    .create_surface = {},
    .requirements = make_requirements(),
  }));

  std::cout << std::format("heap_present: window ready\n");
  if (auto rendered = run_dynamic_rendering(win.ctx()); !rendered) {
    vkexec::examples::abort_with_error(rendered.error());
  }
  std::cout << std::format("heap_present: dynamic rendering ok\n");
  if (run_heap_graphics(win.ctx())) {
    std::cout << std::format("heap_present: heap graphics draw ok\n");
  } else {
    std::cout << std::format("heap_present: skipped heap graphics (extension PFNs unavailable)\n");
  }
  if (run_heap_compute(win.ctx())) {
    std::cout << std::format("heap_present: bindless heap compute ok\n");
  } else {
    std::cout << std::format("heap_present: skipped bindless heap compute (extension PFNs unavailable)\n");
  }

  auto pipeline = vkexec::examples::sync_wait_value(vkexec::factory::graphics_pipeline(
    win.ctx(), win.render_pass(), vkexec::shaders::k_triangle_vert, vkexec::shaders::k_triangle_frag));

  present_frames(win, pipeline);
  win.wait_idle();
  std::cout << std::format("heap_present: completed ({} headless frames)\n", k_present_frames);
  return 0;
}

auto main() -> int { return vkexec::examples::run_example(run); }
