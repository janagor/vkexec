#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/copy.hpp>
#include <vkexec/gpu_buffer.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <memory>
#include <utility>

namespace {

constexpr VkDeviceSize k_bytes = 256;
constexpr std::byte k_marker{ static_cast<unsigned char>(0xAB) };

}// namespace

TEST_CASE("gpu_buffer host_visible is mapped", "[vkexec][gpu_buffer][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto buffer = vkexec::test::sync_wait_value(
    vkexec::gpu_buffer::create(*ctx, k_bytes, vkexec::gpu_buffer_memory::host_visible));
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
  auto ctx = vkexec::test::require_context();
  auto buffer = vkexec::test::sync_wait_value(
    vkexec::gpu_buffer::create(*ctx, k_bytes, vkexec::gpu_buffer_memory::device_local));
  REQUIRE(buffer.handle() != VK_NULL_HANDLE);
  REQUIRE(buffer.size() == k_bytes);
  REQUIRE(buffer.mapped().empty());
}

TEST_CASE("gpu_buffer staging is host-mapped", "[vkexec][gpu_buffer][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto buffer = vkexec::test::sync_wait_value(
    vkexec::gpu_buffer::create(*ctx, k_bytes, vkexec::gpu_buffer_memory::staging));
  REQUIRE(buffer.handle() != VK_NULL_HANDLE);
  REQUIRE(buffer.mapped().size() == static_cast<std::size_t>(k_bytes));
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

  auto ctx = vkexec::test::sync_wait_value(vkexec::context::create({ .requirements = std::move(requirements) }));
  auto buffer = vkexec::test::sync_wait_value(vkexec::gpu_buffer::create(*ctx,
    vkexec::gpu_buffer_create_info{
      .size = k_bytes,
      .memory = vkexec::gpu_buffer_memory::device_local,
      .shader_device_address = true,
    }));
  auto const addr_result = buffer.device_address();
  REQUIRE(addr_result.has_value());
  REQUIRE(*addr_result != 0);
}

TEST_CASE("upload_to_device copies staging bytes into device-local buffer", "[vkexec][gpu_buffer][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto staging = vkexec::test::sync_wait_value(
    vkexec::gpu_buffer::create(*ctx, k_bytes, vkexec::gpu_buffer_memory::staging));
  auto device = vkexec::test::sync_wait_value(
    vkexec::gpu_buffer::create(*ctx, k_bytes, vkexec::gpu_buffer_memory::device_local));

  std::array<std::byte, k_bytes> payload{};
  payload.fill(k_marker);
  REQUIRE(vkexec::upload_to_device(*ctx, staging, device, payload));
}
