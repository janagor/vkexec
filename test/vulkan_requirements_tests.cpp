#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_extensions/descriptor_heap/extension.hpp>
#include <vkexec_extensions/extension.hpp>
#include <vkexec_features/buffer_device_address.hpp>
#include <vkexec_features/dynamic_rendering.hpp>
#include <vkexec_features/feature.hpp>

#include <vulkan/vulkan_core.h>

#include <memory>
#include <utility>

TEST_CASE("context configures Vulkan 1.4 features and descriptor heap extension", "[vkexec][vulkan][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  requirements.optional_device_extensions.push_back(VK_EXT_PRESENT_TIMING_EXTENSION_NAME);
  vkexec::feat::configure<vkexec::feat::dynamic_rendering>(requirements);
  vkexec::ext::configure<vkexec::ext::descriptor_heap>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
  REQUIRE(ctx->device() != VK_NULL_HANDLE);
  REQUIRE(VK_API_VERSION_MAJOR(ctx->api_version()) == 1);
  REQUIRE(VK_API_VERSION_MINOR(ctx->api_version()) == 4);
}

TEST_CASE("context can require bufferDeviceAddress and dynamicRendering", "[vkexec][vulkan][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 3;
  vkexec::feat::configure<vkexec::feat::buffer_device_address>(requirements);
  vkexec::feat::configure<vkexec::feat::dynamic_rendering>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
  REQUIRE(ctx->device() != VK_NULL_HANDLE);
  REQUIRE(VK_API_VERSION_MINOR(ctx->api_version()) >= 3);
  REQUIRE(ctx->procs().get_buffer_device_address != nullptr);
}
