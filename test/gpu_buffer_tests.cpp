#include <catch2/catch_test_macros.hpp>

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/gpu_buffer.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace {

constexpr VkDeviceSize k_bytes = 256;
constexpr std::byte k_marker{ static_cast<unsigned char>(0xAB) };

auto skip_if_no_vulkan(vkexec::error const &err) -> void
{ SKIP(std::string("Vulkan unavailable: ") + std::string(err.message())); }

}// namespace

TEST_CASE("gpu_buffer host_visible is mapped", "[vkexec][gpu_buffer][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto buffer_result = vkexec::gpu_buffer::create(ctx, k_bytes, vkexec::gpu_buffer_memory::host_visible);
  REQUIRE(buffer_result.has_value());
  auto &buffer = *buffer_result;
  REQUIRE(buffer.handle() != VK_NULL_HANDLE);
  REQUIRE(buffer.size() == k_bytes);
  auto const mapped = buffer.mapped();
  REQUIRE(mapped.size() == static_cast<std::size_t>(k_bytes));
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
  mapped[0] = k_marker;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
  REQUIRE(mapped[0] == k_marker);
}

TEST_CASE("gpu_buffer device_local allocates without host mapping", "[vkexec][gpu_buffer][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto buffer_result = vkexec::gpu_buffer::create(ctx, k_bytes, vkexec::gpu_buffer_memory::device_local);
  REQUIRE(buffer_result.has_value());
  auto &buffer = *buffer_result;
  REQUIRE(buffer.handle() != VK_NULL_HANDLE);
  REQUIRE(buffer.size() == k_bytes);
  REQUIRE(buffer.mapped().empty());
}

TEST_CASE("gpu_buffer staging is host-mapped", "[vkexec][gpu_buffer][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto buffer_result = vkexec::gpu_buffer::create(ctx, k_bytes, vkexec::gpu_buffer_memory::staging);
  REQUIRE(buffer_result.has_value());
  auto &buffer = *buffer_result;
  REQUIRE(buffer.handle() != VK_NULL_HANDLE);
  REQUIRE(buffer.mapped().size() == static_cast<std::size_t>(k_bytes));
}

TEST_CASE("gpu_buffer descriptor_heap allocates when extension is available", "[vkexec][gpu_buffer][gpu]")
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

  auto ctx_result = vkexec::context::create({ .requirements = std::move(requirements) });
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto buffer_result = vkexec::gpu_buffer::create(ctx, k_bytes, vkexec::gpu_buffer_memory::descriptor_heap);
  REQUIRE(buffer_result.has_value());
  auto &buffer = *buffer_result;
  REQUIRE(buffer.handle() != VK_NULL_HANDLE);
  REQUIRE(buffer.mapped().size() == static_cast<std::size_t>(k_bytes));
  auto const addr_result = buffer.device_address();
  REQUIRE(addr_result.has_value());
  REQUIRE(*addr_result != 0);
}

TEST_CASE("gpu_buffer device_address works with bufferDeviceAddress enabled", "[vkexec][gpu_buffer][gpu]")
{
  VkPhysicalDeviceVulkan12Features features_12{};
  features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features_12.bufferDeviceAddress = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 2;
  requirements.require_extension_feature(features_12);

  auto ctx_result = vkexec::context::create({ .requirements = std::move(requirements) });
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto buffer_result = vkexec::gpu_buffer::create(ctx,
    vkexec::gpu_buffer_create_info{
      .size = k_bytes,
      .memory = vkexec::gpu_buffer_memory::device_local,
      .shader_device_address = true,
    });
  REQUIRE(buffer_result.has_value());
  auto const addr_result = buffer_result->device_address();
  REQUIRE(addr_result.has_value());
  REQUIRE(*addr_result != 0);
}
