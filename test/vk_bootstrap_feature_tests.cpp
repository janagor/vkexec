#include <vkexec/detail/vk_bootstrap_feature.hpp>

#include <catch2/catch_test_macros.hpp>

#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

#include <utility>

TEST_CASE("private feature adapter retains required and optional requests", "[vkexec][detail]")
{
  VkPhysicalDeviceVulkan12Features required{};
  required.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  required.bufferDeviceAddress = VK_TRUE;

  VkPhysicalDevicePresentTimingFeaturesEXT optional{};
  optional.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT;
  optional.presentTiming = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  vkexec::detail::require_extension_feature(requirements, required);
  vkexec::detail::enable_extension_feature_if_present(requirements, optional);

  auto copy = requirements;
  REQUIRE(copy.required_extension_features.size() == 1);
  REQUIRE(copy.optional_extension_features.size() == 1);

  auto moved = std::move(requirements);
  REQUIRE(moved.required_extension_features.size() == 1);
  REQUIRE(moved.optional_extension_features.size() == 1);
}
