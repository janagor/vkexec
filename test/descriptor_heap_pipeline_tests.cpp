#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/gpu_buffer.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/result.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_extensions/descriptor_heap/algorithm.hpp>
#include <vkexec_extensions/descriptor_heap/buffer.hpp>
#include <vkexec_extensions/descriptor_heap/compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/descriptor_heap.hpp>
#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/heap_graphics_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/resource_table.hpp>
#include <vkexec_extensions/descriptor_heap/strategy.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/triangle_shaders.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

namespace ex = stdexec;

namespace {

constexpr std::uint32_t k_work_count = 64;
constexpr VkDeviceSize k_storage_bytes = 256;

constexpr std::string_view k_heap_compute_glsl = R"(#version 460
layout(local_size_x = 64) in;
void main() {}
)";

constexpr std::string_view k_heap_spec_glsl = R"(#version 460
layout(local_size_x = 64) in;
layout(constant_id = 0) const uint kCount = 1;
void main() {
  if (gl_GlobalInvocationID.x >= kCount) { return; }
}
)";

struct heap_push
{
  std::uint32_t count;
};

}// namespace

TEST_CASE("compute_pipeline can create a descriptor-heap null layout", "[vkexec][descriptor_heap][gpu]")
{
  VkPhysicalDeviceDescriptorHeapFeaturesEXT features_heap{};
  features_heap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
  features_heap.descriptorHeap = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  requirements.require_extension_feature(features_heap);

  auto ctx = vkexec::test::sync_wait_value(vkexec::context::create({ .requirements = std::move(requirements) }));
  auto pipe = vkexec::test::sync_wait_value(vkexec::compute_pipeline::create(vkexec::descriptor_heap,
    *ctx,
    k_heap_compute_glsl,
    vkexec::heap_layout_desc{ .specialization = {}, .local_size = vkexec::k_default_local_size },
    "heap.comp"));
  REQUIRE(pipe.resources().pipeline != VK_NULL_HANDLE);
  REQUIRE(pipe.resources().pipeline_layout == VK_NULL_HANDLE);
  REQUIRE(pipe.resources().set_layout == VK_NULL_HANDLE);
  REQUIRE(pipe.resources().descriptor_pool == VK_NULL_HANDLE);
}

TEST_CASE("descriptor-heap compute_pipeline accepts specialization constants", "[vkexec][descriptor_heap][gpu]")
{
  VkPhysicalDeviceDescriptorHeapFeaturesEXT features_heap{};
  features_heap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
  features_heap.descriptorHeap = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  requirements.require_extension_feature(features_heap);

  auto ctx = vkexec::test::sync_wait_value(vkexec::context::create({ .requirements = std::move(requirements) }));
  auto pipe = vkexec::test::sync_wait_value(vkexec::compute_pipeline::create(vkexec::descriptor_heap,
    *ctx,
    k_heap_spec_glsl,
    vkexec::heap_layout_desc{ .specialization = { k_work_count }, .local_size = vkexec::k_default_local_size },
    "heap_spec.comp"));
  REQUIRE(pipe.resources().pipeline != VK_NULL_HANDLE);
}

TEST_CASE("compute_pass records push data for descriptor-heap pipelines", "[vkexec][descriptor_heap][gpu]")
{
  VkPhysicalDeviceDescriptorHeapFeaturesEXT features_heap{};
  features_heap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
  features_heap.descriptorHeap = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  requirements.require_extension_feature(features_heap);

  auto ctx = vkexec::test::sync_wait_value(vkexec::context::create({ .requirements = std::move(requirements) }));
  auto pipe = vkexec::test::sync_wait_value(vkexec::compute_pipeline::create(vkexec::descriptor_heap,
    *ctx,
    k_heap_compute_glsl,
    vkexec::heap_layout_desc{ .specialization = {}, .local_size = vkexec::k_default_local_size },
    "heap_pass.comp"));

  heap_push const params{ .count = k_work_count };
  auto waited = vkexec::test::sync_wait_sender(
    ex::schedule(ctx->get_scheduler()) | vkexec::compute_pass(vkexec::descriptor_heap, pipe, params, k_work_count));
  REQUIRE(vkexec::test::sync_wait_completed(waited));
}

TEST_CASE("dispatch_compute builds descriptor-heap algorithm passes", "[vkexec][descriptor_heap][gpu]")
{
  VkPhysicalDeviceDescriptorHeapFeaturesEXT features_heap{};
  features_heap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
  features_heap.descriptorHeap = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  requirements.require_extension_feature(features_heap);

  auto ctx = vkexec::test::sync_wait_value(vkexec::context::create({ .requirements = std::move(requirements) }));
  vkexec::algorithm const algo = vkexec::test::sync_wait_value(vkexec::compute_pipeline::create(vkexec::descriptor_heap,
    *ctx,
    k_heap_compute_glsl,
    vkexec::heap_layout_desc{ .specialization = {}, .local_size = vkexec::k_default_local_size },
    "heap_dispatch.comp"));

  heap_push const params{ .count = k_work_count };
  auto waited = vkexec::test::sync_wait_sender(
    ex::schedule(ctx->get_scheduler()) | vkexec::dispatch_compute(vkexec::descriptor_heap, algo, params, k_work_count));
  REQUIRE(vkexec::test::sync_wait_completed(waited));
}

TEST_CASE("tagged create_compute_resources draws without owning pipeline", "[vkexec][descriptor_heap][gpu][execution]")
{
  VkPhysicalDeviceVulkan12Features features_12{};
  features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features_12.bufferDeviceAddress = VK_TRUE;

  VkPhysicalDeviceDescriptorHeapFeaturesEXT features_heap{};
  features_heap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
  features_heap.descriptorHeap = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  requirements.require_extension_feature(features_12).require_extension_feature(features_heap);

  auto ctx = vkexec::test::sync_wait_value(vkexec::context::create({ .requirements = std::move(requirements) }));
  auto layout_result = vkexec::query_descriptor_heap_layout(*ctx);
  REQUIRE(layout_result.has_value());
  auto const &layout = vkexec::expected_get(layout_result);
  auto storage = vkexec::test::sync_wait_value(vkexec::gpu_buffer::create(*ctx,
    vkexec::gpu_buffer_create_info{
      .size = k_storage_bytes,
      .memory = vkexec::gpu_buffer_memory::device_local,
      .shader_device_address = true,
    }));
  auto heap = vkexec::test::sync_wait_value(
    vkexec::descriptor_heap_buffer::create(*ctx, vkexec::descriptor_heap_byte_size(layout, 1)));
  auto const table = vkexec::bindings(
    vkexec::resource_binding{ .slot = 0, .resource = vkexec::buffer_resource(storage.handle(), storage.size()) });
  std::array<std::uint32_t, 1> const indices{ 0 };
  vkexec::heap_table_lower_env const lower_env{ .resource_heap_bytes = heap.mapped(),
    .sampler_heap_bytes = {},
    .buffer_descriptor_size = layout.buffer_descriptor_size,
    .image_descriptor_size = layout.image_descriptor_size,
    .descriptor_stride = layout.descriptor_stride,
    .sampler_descriptor_size = layout.sampler_descriptor_size,
    .sampler_descriptor_stride = layout.sampler_descriptor_size,
    .indices = indices,
    .sampler_indices = {},
    .image_view_infos = {},
    .sampler_infos = {} };
  auto resources_result = vkexec::create_compute_resources(vkexec::descriptor_heap,
    *ctx,
    k_heap_compute_glsl,
    vkexec::heap_layout_desc{ .specialization = {}, .local_size = vkexec::k_default_local_size },
    "heap_execution.comp");
  REQUIRE(resources_result.has_value());
  auto resources = vkexec::expected_take(resources_result);
  REQUIRE(resources.pipeline != VK_NULL_HANDLE);
  REQUIRE(resources.pipeline_layout == VK_NULL_HANDLE);
  REQUIRE(resources.set_layout == VK_NULL_HANDLE);
  REQUIRE(resources.descriptor_pool == VK_NULL_HANDLE);

  heap_push const params{ .count = k_work_count };
  auto const groups = vkexec::groups_for(resources, k_work_count);
  auto graph = ex::schedule(ctx->get_scheduler())
               | vkexec::bind_resources(vkexec::descriptor_heap, resources, table, lower_env, params)
               | vkexec::compute_pass(vkexec::bind_compute(resources), groups);
  auto waited = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(waited));

  vkexec::destroy_compute_resources(*ctx, resources);
}

TEST_CASE("create_graphics_resources builds null-layout DR pipeline", "[vkexec][descriptor_heap][gpu]")
{
  VkPhysicalDeviceDescriptorHeapFeaturesEXT features_heap{};
  features_heap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
  features_heap.descriptorHeap = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  requirements.require_extension_feature(features_heap);

  auto ctx = vkexec::test::sync_wait_value(vkexec::context::create({ .requirements = std::move(requirements) }));
  auto resources_result = vkexec::create_graphics_resources(vkexec::descriptor_heap,
    *ctx,
    vkexec::shaders::k_triangle_vert,
    vkexec::shaders::k_triangle_frag,
    vkexec::heap_graphics_layout_desc{
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .blend = vkexec::blend_mode::premultiplied,
      .depth_test = false,
      .depth_write = false,
      .color_formats = { VK_FORMAT_R8G8B8A8_UNORM },
      .depth_format = VK_FORMAT_UNDEFINED,
    },
    "heap_gfx.vert",
    "heap_gfx.frag");
  REQUIRE(resources_result.has_value());
  auto resources = vkexec::expected_take(resources_result);
  REQUIRE(resources.pipeline != VK_NULL_HANDLE);
  REQUIRE(resources.pipeline_layout == VK_NULL_HANDLE);
  REQUIRE(resources.set_layout == VK_NULL_HANDLE);
  REQUIRE(resources.descriptor_pool == VK_NULL_HANDLE);
  REQUIRE(resources.shader == VK_NULL_HANDLE);

  vkexec::destroy_graphics_resources(vkexec::descriptor_heap, *ctx, resources);
  REQUIRE(resources.pipeline == VK_NULL_HANDLE);
}

TEST_CASE("graphics_pipeline tag factory owns null-layout DR pipeline", "[vkexec][descriptor_heap][gpu]")
{
  VkPhysicalDeviceDescriptorHeapFeaturesEXT features_heap{};
  features_heap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
  features_heap.descriptorHeap = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  requirements.require_extension_feature(features_heap);

  auto ctx = vkexec::test::sync_wait_value(vkexec::context::create({ .requirements = std::move(requirements) }));
  auto pipe = vkexec::test::sync_wait_value(vkexec::graphics_pipeline::create(vkexec::descriptor_heap,
    *ctx,
    vkexec::shaders::k_triangle_vert,
    vkexec::shaders::k_triangle_frag,
    vkexec::heap_graphics_layout_desc{
      .blend = vkexec::blend_mode::premultiplied,
      .depth_test = false,
      .depth_write = false,
      .color_formats = { VK_FORMAT_R8G8B8A8_UNORM },
    },
    "heap_own.vert",
    "heap_own.frag"));
  REQUIRE(pipe.resources().pipeline != VK_NULL_HANDLE);
  REQUIRE(pipe.bind().pipeline == pipe.resources().pipeline);
  REQUIRE(pipe.bind().layout == VK_NULL_HANDLE);
  REQUIRE(pipe.bind().set == VK_NULL_HANDLE);
}
