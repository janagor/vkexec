#include "test_helpers.hpp"
#include "vma_test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/buffer.hpp>
#include <vkexec/sync_wait.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace {

constexpr std::size_t k_count = 8;
constexpr float k_fill = 2.5F;
constexpr std::uint32_t k_int_fill = 7U;

}// namespace

TEST_CASE("factory::make_buffer sender completes with a filled buffer", "[vkexec][buffer][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto values = vkexec::test::sync_wait_value(vkexec::factory::make_buffer(allocator, k_count, k_fill));
  REQUIRE(values.size() == k_count);
  REQUIRE(values.vk_buffer() != VK_NULL_HANDLE);
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  REQUIRE(values.data()[0] == k_fill);
  REQUIRE(values.data()[k_count - 1] == k_fill);
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

TEST_CASE("factory::make_buffer works for integral element types", "[vkexec][buffer][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto values = vkexec::test::sync_wait_value(vkexec::factory::make_buffer(allocator, k_count, k_int_fill));
  REQUIRE(values.size() == k_count);
}

TEST_CASE("sync_wait_value completes factory::make_buffer", "[vkexec][buffer][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto values = vkexec::sync_wait_value(vkexec::factory::make_buffer(allocator, k_count, k_fill));
  REQUIRE(values.size() == k_count);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  REQUIRE(values.data()[0] == k_fill);
}

TEST_CASE("try_sync_wait_value completes factory::make_buffer", "[vkexec][buffer][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto values = vkexec::try_sync_wait_value(vkexec::factory::make_buffer(allocator, k_count, k_fill));
  REQUIRE(values.has_value());
  REQUIRE(values->size() == k_count);
}
