#include <catch2/catch_test_macros.hpp>

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

#include <memory>
#include <string>
#include <utility>

namespace {

auto skip_if_unavailable(vkexec::error const &err) -> void
{ SKIP(std::string("Vulkan feature set unavailable: ") + std::string(err.message())); }

}// namespace

TEST_CASE("context can require Vulkan 1.4 features and extension feature structs", "[vkexec][vulkan][gpu]")
{
  VkPhysicalDeviceVulkan12Features features_12{};
  features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features_12.bufferDeviceAddress = VK_TRUE;

  VkPhysicalDeviceVulkan13Features features_13{};
  features_13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  features_13.dynamicRendering = VK_TRUE;

  VkPhysicalDeviceDescriptorHeapFeaturesEXT features_heap{};
  features_heap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
  features_heap.descriptorHeap = VK_TRUE;

  VkPhysicalDevicePresentTimingFeaturesEXT features_present_timing{};
  features_present_timing.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT;
  features_present_timing.presentTiming = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  requirements.optional_device_extensions = { VK_EXT_PRESENT_TIMING_EXTENSION_NAME };
  requirements.require_extension_feature(features_12)
    .require_extension_feature(features_13)
    .require_extension_feature(features_heap)
    .enable_extension_feature_if_present(features_present_timing);

  auto ctx_result = vkexec::context::create({ .requirements = std::move(requirements) });
  if (!ctx_result) { skip_if_unavailable(ctx_result.error()); }

  auto &ctx = **ctx_result;
  REQUIRE(ctx.device() != VK_NULL_HANDLE);
  REQUIRE(VK_API_VERSION_MAJOR(ctx.api_version()) == 1);
  REQUIRE(VK_API_VERSION_MINOR(ctx.api_version()) == 4);
  REQUIRE(ctx.procs().write_resource_descriptors != nullptr);
  REQUIRE(ctx.procs().cmd_bind_resource_heap != nullptr);
  REQUIRE(ctx.procs().cmd_push_data != nullptr);
}

TEST_CASE("context can require bufferDeviceAddress and dynamicRendering", "[vkexec][vulkan][gpu]")
{
  VkPhysicalDeviceVulkan12Features features_12{};
  features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features_12.bufferDeviceAddress = VK_TRUE;

  VkPhysicalDeviceVulkan13Features features_13{};
  features_13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  features_13.dynamicRendering = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 3;
  requirements.require_extension_feature(features_12).require_extension_feature(features_13);

  auto ctx_result = vkexec::context::create({ .requirements = std::move(requirements) });
  if (!ctx_result) { skip_if_unavailable(ctx_result.error()); }

  auto &ctx = **ctx_result;
  REQUIRE(ctx.device() != VK_NULL_HANDLE);
  REQUIRE(VK_API_VERSION_MINOR(ctx.api_version()) >= 3);
  REQUIRE(ctx.procs().get_buffer_device_address != nullptr);
}
