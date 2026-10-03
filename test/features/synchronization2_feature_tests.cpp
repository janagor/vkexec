#include <catch2/catch_test_macros.hpp>

#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_features/feature.hpp>
#include <vkexec_features/synchronization2.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstring>

TEST_CASE("Vulkan 1.0 synchronization2 requirement includes instance dependency", "[vkexec][sync]")
{
  vkexec::vulkan_requirements requirements{};
  vkexec::feat::configure<vkexec::feat::synchronization2>(requirements);
  REQUIRE(std::ranges::any_of(requirements.instance_extensions, [](char const *extension) -> bool {
    return std::strcmp(extension, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == 0;
  }));
  REQUIRE(std::ranges::any_of(requirements.device_extensions, [](char const *extension) -> bool {
    return std::strcmp(extension, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0;
  }));
  REQUIRE(requirements.required_extension_features.size() == 1);
}

TEST_CASE("Vulkan 1.2 synchronization2 requires only the KHR device extension", "[vkexec][sync]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_minor = 2;
  vkexec::feat::configure<vkexec::feat::synchronization2>(requirements);
  REQUIRE(requirements.instance_extensions.empty());
  REQUIRE(std::ranges::any_of(requirements.device_extensions, [](char const *extension) -> bool {
    return std::strcmp(extension, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0;
  }));
  REQUIRE(requirements.required_extension_features.size() == 1);
}

TEST_CASE("Vulkan 1.3 synchronization2 requires no KHR extensions", "[vkexec][sync]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_minor = 3;
  vkexec::feat::configure<vkexec::feat::synchronization2>(requirements);
  REQUIRE(requirements.instance_extensions.empty());
  REQUIRE(requirements.device_extensions.empty());
  REQUIRE(requirements.required_extension_features.size() == 1);
}
