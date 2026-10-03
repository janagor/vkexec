#include <catch2/catch_test_macros.hpp>

#include <vkexec/error.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_extensions/descriptor_heap/buffer.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <concepts>
#include <cstddef>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr VkDeviceSize k_heap_bytes = 64;

struct test_buffer
{
  std::vector<std::byte> bytes;

  [[nodiscard]] static auto handle() noexcept -> VkBuffer { return VK_NULL_HANDLE; }
  [[nodiscard]] auto size() const noexcept -> VkDeviceSize { return bytes.size(); }
  [[nodiscard]] auto mapped() noexcept -> std::span<std::byte> { return bytes; }
  [[nodiscard]] auto mapped() const noexcept -> std::span<std::byte const> { return bytes; }
  [[nodiscard]] static auto flush() -> vkexec::status { return {}; }
  [[nodiscard]] static auto device_address() -> vkexec::result<VkDeviceAddress> { return VkDeviceAddress{}; }
};

struct test_allocator
{
  using buffer_type = test_buffer;
  int allocations{};
  vkexec::buffer_create_info last_info{};
};

[[nodiscard]] auto
  tag_invoke(vkexec::allocate_buffer_t /*tag*/, test_allocator &allocator, vkexec::buffer_create_info info)
{
  return vkexec::make_sender([&allocator, info]() -> vkexec::result<test_buffer> {
    ++allocator.allocations;
    allocator.last_info = info;
    return test_buffer{ std::vector<std::byte>(static_cast<std::size_t>(info.size)) };
  });
}

using heap_sender =
  decltype(vkexec::factory::make_descriptor_heap_buffer(std::declval<test_allocator &>(), k_heap_bytes));
static_assert(stdexec::sender<heap_sender>);
// NOLINTBEGIN(misc-include-cleaner)
static_assert(
  std::same_as<stdexec::error_types_of_t<heap_sender, stdexec::env<>, std::tuple>, std::tuple<vkexec::error>>);
// NOLINTEND(misc-include-cleaner)

}// namespace

TEST_CASE("descriptor heap buffer uses a custom allocator", "[vkexec][descriptor_heap][resource_allocator]")
{
  test_allocator allocator;
  auto heap = vkexec::try_sync_wait_value(vkexec::factory::make_descriptor_heap_buffer(allocator, k_heap_bytes));
  REQUIRE(heap.has_value());
  REQUIRE(allocator.allocations == 1);
  REQUIRE(allocator.last_info.alignment == vkexec::k_descriptor_heap_buffer_alignment);
  REQUIRE(heap->mapped().size() == k_heap_bytes);
}
