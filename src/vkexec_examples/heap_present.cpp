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
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_edsl/types.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/window.hpp>

#include <stdexec/execution.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <exception>
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

auto run_dynamic_rendering(vkexec::context &ctx) -> void
{
  auto img = vkexec::image::create(ctx,
    vkexec::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::image_usage::color_storage,
    });
  auto view = vkexec::image_view::create(ctx, img);

  VkCommandBuffer cmd = ctx.allocate_command_buffer();
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
    throw std::runtime_error("vkBeginCommandBuffer failed (dynamic rendering)");
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
  color.clear.color = { { 0.15F, 0.35F, 0.55F, 1.0F } };
  std::array<vkexec::color_attachment, 1> const colors{ color };
  vkexec::cmd_begin_rendering(cmd,
    vkexec::rendering_info{
      .extent = img.extent(),
      .color = colors,
    });
  vkexec::cmd_end_rendering(cmd);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) { throw std::runtime_error("vkEndCommandBuffer failed"); }
  std::array<VkCommandBuffer, 1> const cmds{ cmd };

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence{ VK_NULL_HANDLE };
  if (vkCreateFence(ctx.device(), &fence_info, nullptr, &fence) != VK_SUCCESS) {
    throw std::runtime_error("vkCreateFence failed (dynamic rendering)");
  }

  ctx.submit(vkexec::queue_submit{
    .command_buffers = cmds,
    .fence = fence,
    .queue = ctx.graphics_queue() != VK_NULL_HANDLE ? ctx.graphics_queue() : ctx.compute_queue(),
  });
  if (vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
    vkDestroyFence(ctx.device(), fence, nullptr);
    throw std::runtime_error("vkWaitForFences failed (dynamic rendering)");
  }
  vkDestroyFence(ctx.device(), fence, nullptr);
  ctx.free_command_buffer(cmd);
}

auto run_heap_compute(vkexec::context &ctx) -> bool
{
  if (ctx.procs().cmd_push_data == nullptr || ctx.procs().write_resource_descriptors == nullptr) { return false; }

  auto const layout = vkexec::query_descriptor_heap_layout(ctx);
  auto storage = vkexec::gpu_buffer::create(ctx,
    vkexec::gpu_buffer_create_info{
      .size = k_storage_bytes,
      .memory = vkexec::gpu_buffer_memory::device_local,
      .shader_device_address = true,
    });
  auto heap = vkexec::gpu_buffer::create(ctx,
    vkexec::descriptor_heap_byte_size(layout, k_heap_slots),
    vkexec::gpu_buffer_memory::descriptor_heap);

  auto mapped = heap.mapped();
  vkexec::write_storage_buffer_descriptor(
    ctx, storage.device_address(), storage.size(), mapped.subspan(0, layout.buffer_descriptor_size));

  auto pipe = vkexec::compute_pipeline::from_glsl(ctx,
    k_heap_glsl,
    vkexec::layout_desc{
      .bindings = {},
      .push_constant_size = sizeof(heap_push),
      .specialization = {},
      .local_size = { 64, 1, 1 },
      .descriptor_heap = true,
    },
    "heap_present.comp");
  if (!pipe.has_value()) { throw std::runtime_error(std::string(pipe.error().message())); }

  heap_push const params{ .count = 64 };
  auto result = ex::sync_wait(ex::schedule(ctx.get_scheduler()) | vkexec::compute_pass(*pipe, params, 64U));
  if (!result.has_value()) { throw std::runtime_error("bindless compute_pass failed"); }
  return true;
}

auto present_frames(vkexec::window &win, vkexec::graphics_pipeline &pipeline) -> void
{
  for (std::uint32_t frame = 0; frame < k_present_frames; ++frame) {
    (void)ex::sync_wait(
      ex::schedule(win.ctx().get_scheduler()) | vkexec::draw(win, pipeline, k_triangle_vertices) | vkexec::submit);
  }
}

}// namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
auto main() -> int
{
  try {
    auto win = vkexec::window::headless(vkexec::window::config{
      .width = k_width,
      .height = k_height,
      .title = "vkexec heap_present",
      .headless = true,
      .requirements = make_requirements(),
    });

    std::println("heap_present: window ready");
    run_dynamic_rendering(win.ctx());
    std::println("heap_present: dynamic rendering ok");
    if (run_heap_compute(win.ctx())) {
      std::println("heap_present: bindless heap compute ok");
    } else {
      std::println("heap_present: skipped bindless heap compute (extension PFNs unavailable)");
    }

    vkexec::graphics_pipeline pipeline(
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

    present_frames(win, pipeline);
    win.wait_idle();
    std::println("heap_present: completed ({} headless frames)", k_present_frames);
    return 0;
  } catch (std::exception const &ex) {
    std::println(stderr, "vkexec heap_present example failed: {}", ex.what());
    return 1;
  }
}
