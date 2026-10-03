#include "test_helpers.hpp"
#include "vma_test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec_vma/copy.hpp>
#include <vkexec_vma/gpu_buffer.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>

namespace {

constexpr VkDeviceSize k_bytes = 256;
constexpr std::byte k_marker{ static_cast<unsigned char>(0xAB) };

}// namespace

TEST_CASE("gpu_buffer host_visible is mapped", "[vkexec][gpu_buffer][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto buffer = vkexec::test::sync_wait_value(
    vkexec::vma::factory::make_gpu_buffer(allocator, k_bytes, vkexec::vma::buffer_memory::host_visible));
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
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto buffer = vkexec::test::sync_wait_value(
    vkexec::vma::factory::make_gpu_buffer(allocator, k_bytes, vkexec::vma::buffer_memory::device_local));
  REQUIRE(buffer.handle() != VK_NULL_HANDLE);
  REQUIRE(buffer.size() == k_bytes);
  REQUIRE(buffer.mapped().empty());
}

TEST_CASE("gpu_buffer staging is host-mapped", "[vkexec][gpu_buffer][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto buffer = vkexec::test::sync_wait_value(
    vkexec::vma::factory::make_gpu_buffer(allocator, k_bytes, vkexec::vma::buffer_memory::staging));
  REQUIRE(buffer.handle() != VK_NULL_HANDLE);
  REQUIRE(buffer.mapped().size() == static_cast<std::size_t>(k_bytes));
}

TEST_CASE("upload_to_device copies staging bytes into device-local buffer", "[vkexec][gpu_buffer][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto staging = vkexec::test::sync_wait_value(
    vkexec::vma::factory::make_gpu_buffer(allocator, k_bytes, vkexec::vma::buffer_memory::staging));
  auto device = vkexec::test::sync_wait_value(
    vkexec::vma::factory::make_gpu_buffer(allocator, k_bytes, vkexec::vma::buffer_memory::device_local));

  std::array<std::byte, k_bytes> payload{};
  payload.fill(k_marker);
  REQUIRE(vkexec::vma::upload_to_device(*ctx, staging, device, payload));
}
