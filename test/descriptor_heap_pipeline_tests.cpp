#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_extensions/descriptor_heap/compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

namespace ex = stdexec;

namespace {

constexpr std::uint32_t k_work_count = 64;

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

TEST_CASE("heap_compute_pipeline can create a descriptor-heap null layout", "[vkexec][descriptor_heap][gpu]")
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
  auto pipe = vkexec::test::sync_wait_value(vkexec::heap_compute_pipeline::create(*ctx,
    k_heap_compute_glsl,
    vkexec::heap_layout_desc{ .specialization = {}, .local_size = vkexec::k_default_local_size },
    "heap.comp"));
  REQUIRE(pipe.resources().pipeline != VK_NULL_HANDLE);
  REQUIRE(pipe.resources().pipeline_layout == VK_NULL_HANDLE);
  REQUIRE(pipe.resources().set_layout == VK_NULL_HANDLE);
  REQUIRE(pipe.resources().descriptor_pool == VK_NULL_HANDLE);
}

TEST_CASE("heap_compute_pipeline accepts specialization constants", "[vkexec][descriptor_heap][gpu]")
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
  auto pipe = vkexec::test::sync_wait_value(vkexec::heap_compute_pipeline::create(*ctx,
    k_heap_spec_glsl,
    vkexec::heap_layout_desc{ .specialization = { k_work_count }, .local_size = vkexec::k_default_local_size },
    "heap_spec.comp"));
  REQUIRE(pipe.resources().pipeline != VK_NULL_HANDLE);
}

TEST_CASE("compute_heap_pass records bindless push data for heap pipelines", "[vkexec][descriptor_heap][gpu]")
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
  auto pipe = vkexec::test::sync_wait_value(vkexec::heap_compute_pipeline::create(*ctx,
    k_heap_compute_glsl,
    vkexec::heap_layout_desc{ .specialization = {}, .local_size = vkexec::k_default_local_size },
    "heap_pass.comp"));

  heap_push const params{ .count = k_work_count };
  auto waited = vkexec::test::sync_wait_sender(
    ex::schedule(ctx->get_scheduler()) | vkexec::compute_heap_pass(pipe, params, k_work_count));
  REQUIRE(vkexec::test::sync_wait_completed(waited));
}

TEST_CASE("create_heap_compute_resources draws without owning pipeline", "[vkexec][descriptor_heap][gpu][execution]")
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
  auto resources_result = vkexec::create_heap_compute_resources(*ctx,
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
  auto waited = vkexec::test::sync_wait_sender(ex::schedule(ctx->get_scheduler())
                                               | vkexec::compute_heap_pass(
                                                 vkexec::bind_heap(resources), params, vkexec::groups_for(resources, k_work_count)));
  REQUIRE(vkexec::test::sync_wait_completed(waited));

  vkexec::destroy_heap_compute_resources(*ctx, resources);
}
