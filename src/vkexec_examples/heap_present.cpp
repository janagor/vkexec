#include <vkexec/barrier.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/descriptor_heap.hpp>
#include <vkexec/gpu_buffer.hpp>
#include <vkexec/image.hpp>
#include <vkexec/image_view.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/rendering.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_edsl/types.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/window.hpp>

#include <stdexec/execution.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <memory>
#include <print>
#include <string>
#include <string_view>
#include <utility>

namespace ex = stdexec;
namespace edsl = vkexec::edsl;

namespace {

constexpr std::uint32_t k_width = 64;
constexpr std::uint32_t k_height = 64;
constexpr std::uint32_t k_present_frames = 3;
constexpr std::uint32_t k_triangle_vertices = 3;
constexpr std::size_t k_heap_slots = 1;
constexpr VkDeviceSize k_storage_bytes = 256;
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
  VkPhysicalDeviceVulkan12Features features_12{};
  features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features_12.bufferDeviceAddress = VK_TRUE;
  features_12.timelineSemaphore = VK_TRUE;

  VkPhysicalDeviceVulkan13Features features_13{};
  features_13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  features_13.dynamicRendering = VK_TRUE;

  VkPhysicalDeviceDescriptorHeapFeaturesEXT features_heap{};
  features_heap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
  features_heap.descriptorHeap = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 3;
  requirements.optional_device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  requirements.require_extension_feature(features_12)
    .require_extension_feature(features_13)
    .enable_extension_feature_if_present(features_heap);
  return requirements;
}

auto run_dynamic_rendering(vkexec::context &ctx) -> vkexec::status
{
  auto img_result = vkexec::image::create(ctx,
    vkexec::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::image_usage::color_storage,
    });
  if (!img_result) { return std::unexpected(img_result.error()); }
  auto view_result = vkexec::image_view::create(ctx, *img_result);
  if (!view_result) { return std::unexpected(view_result.error()); }

  auto cmd_result = ctx.allocate_command_buffer();
  if (!cmd_result) { return std::unexpected(cmd_result.error()); }
  VkCommandBuffer cmd = *cmd_result;
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
    return std::unexpected(vkexec::make_vk_error(VK_ERROR_UNKNOWN, "vkBeginCommandBuffer failed (dynamic rendering)"));
  }

  vkexec::image_barrier(cmd,
    {
      .image = img_result->handle(),
      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
      .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      .dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .src_access = 0,
      .dst_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    });

  vkexec::color_attachment color{};
  color.view = view_result->handle();
  color.clear.color = { { 0.15F, 0.35F, 0.55F, 1.0F } };
  std::array<vkexec::color_attachment, 1> const colors{ color };
  if (auto const began = vkexec::cmd_begin_rendering(cmd,
        vkexec::rendering_info{
          .extent = img_result->extent(),
          .color = colors,
        });
      !began) {
    ctx.free_command_buffer(cmd);
    return began;
  }
  if (auto const ended = vkexec::cmd_end_rendering(cmd); !ended) {
    ctx.free_command_buffer(cmd);
    return ended;
  }

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    ctx.free_command_buffer(cmd);
    return std::unexpected(vkexec::make_vk_error(VK_ERROR_UNKNOWN, "vkEndCommandBuffer failed"));
  }
  std::array<VkCommandBuffer, 1> const cmds{ cmd };

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence{ VK_NULL_HANDLE };
  if (vkCreateFence(ctx.device(), &fence_info, nullptr, &fence) != VK_SUCCESS) {
    ctx.free_command_buffer(cmd);
    return std::unexpected(vkexec::make_vk_error(VK_ERROR_UNKNOWN, "vkCreateFence failed (dynamic rendering)"));
  }

  if (auto const submitted = ctx.submit(vkexec::queue_submit{
        .command_buffers = cmds,
        .fence = fence,
        .queue = ctx.graphics_queue() != VK_NULL_HANDLE ? ctx.graphics_queue() : ctx.compute_queue(),
      });
      !submitted) {
    vkDestroyFence(ctx.device(), fence, nullptr);
    ctx.free_command_buffer(cmd);
    return submitted;
  }
  if (vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
    vkDestroyFence(ctx.device(), fence, nullptr);
    ctx.free_command_buffer(cmd);
    return std::unexpected(vkexec::make_vk_error(VK_ERROR_UNKNOWN, "vkWaitForFences failed (dynamic rendering)"));
  }
  vkDestroyFence(ctx.device(), fence, nullptr);
  ctx.free_command_buffer(cmd);
  return {};
}

auto run_heap_compute(vkexec::context &ctx) -> bool
{
  if (ctx.procs().cmd_push_data == nullptr || ctx.procs().write_resource_descriptors == nullptr) { return false; }

  auto const layout_result = vkexec::query_descriptor_heap_layout(ctx);
  if (!layout_result) { return false; }
  auto const &layout = *layout_result;
  auto storage_result = vkexec::gpu_buffer::create(ctx,
    vkexec::gpu_buffer_create_info{
      .size = k_storage_bytes,
      .memory = vkexec::gpu_buffer_memory::device_local,
      .shader_device_address = true,
    });
  if (!storage_result) { return false; }
  auto heap_result = vkexec::gpu_buffer::create(ctx,
    vkexec::descriptor_heap_byte_size(layout, k_heap_slots),
    vkexec::gpu_buffer_memory::descriptor_heap);
  if (!heap_result) { return false; }

  auto const storage_addr = storage_result->device_address();
  if (!storage_addr) { return false; }
  auto mapped = heap_result->mapped();
  if (auto const written = vkexec::write_storage_buffer_descriptor(
        ctx, *storage_addr, storage_result->size(), mapped.subspan(0, layout.buffer_descriptor_size));
      !written) {
    return false;
  }

  auto pipe = vkexec::compute_pipeline::create(ctx,
    k_heap_glsl,
    vkexec::layout_desc{
      .bindings = {},
      .push_constant_size = 0,
      .specialization = {},
      .local_size = { 64, 1, 1 },
      .descriptor_heap = true,
    },
    "heap_present.comp");
  if (!pipe) { return false; }

  heap_push const params{ .count = 64 };
  auto waited = vkexec::sync_wait(ex::schedule(ctx.get_scheduler()) | vkexec::compute_pass(*pipe, params, 64U));
  return waited.has_value() && waited->has_value();
}

auto present_frames(vkexec::window &win, vkexec::graphics_pipeline &pipeline) -> void
{
  for (std::uint32_t frame = 0; frame < k_present_frames; ++frame) {
    (void)vkexec::sync_wait(
      ex::schedule(win.ctx().get_scheduler()) | vkexec::draw(win, pipeline, k_triangle_vertices) | vkexec::submit);
  }
}

}// namespace

auto main() -> int
{
  auto win_result = vkexec::window::headless(vkexec::window::config{
    .width = k_width,
    .height = k_height,
    .title = "vkexec heap_present",
    .headless = true,
    .requirements = make_requirements(),
  });
  if (!win_result) {
    std::println(stderr, "vkexec heap_present example failed: {}", win_result.error().message());
    return 1;
  }
  auto win = std::move(*win_result);

  std::println("heap_present: window ready");
  if (auto const rendered = run_dynamic_rendering(win.ctx()); !rendered) {
    std::println(stderr, "vkexec heap_present example failed: {}", rendered.error().message());
    return 1;
  }
  std::println("heap_present: dynamic rendering ok");
  if (run_heap_compute(win.ctx())) {
    std::println("heap_present: bindless heap compute ok");
  } else {
    std::println("heap_present: skipped bindless heap compute (extension PFNs unavailable)");
  }

  auto pipeline_result = vkexec::graphics_pipeline::create(
    win.ctx(),
    win.render_pass(),
    [](edsl::Int vertex_id, edsl::VertexWriter out) -> void {
      edsl::Float2 const pos = edsl::select(vertex_id == edsl::Int::constant(0),
        edsl::vec2(0.0, -0.5),
        edsl::select(vertex_id == edsl::Int::constant(1), edsl::vec2(0.5, 0.5), edsl::vec2(-0.5, 0.5)));
      edsl::Float3 const col = edsl::select(vertex_id == edsl::Int::constant(0),
        edsl::vec3(1.0, 0.2, 0.2),
        edsl::select(vertex_id == edsl::Int::constant(1), edsl::vec3(0.2, 1.0, 0.2), edsl::vec3(0.2, 0.4, 1.0)));
      out.position(pos);
      out.color(col);
    },
    [](edsl::FragmentReader fragment_in, edsl::FragmentWriter out) -> void {
      out.color(edsl::vec4(fragment_in.color(), 1.0));
    });
  if (!pipeline_result) {
    std::println(stderr, "vkexec heap_present example failed: {}", pipeline_result.error().message());
    return 1;
  }

  present_frames(win, *pipeline_result);
  win.wait_idle();
  std::println("heap_present: completed ({} headless frames)", k_present_frames);
  return 0;
}
