#include <catch2/catch_test_macros.hpp>

#include <vkexec/pipeline.hpp>
#include <vkexec/submit_scope.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <span>

namespace {

constexpr VkDeviceSize k_byte_size = 64;

[[nodiscard]] auto fake_buffer(void *storage) -> VkBuffer { return static_cast<VkBuffer>(storage); }

}// namespace

TEST_CASE("storage_bindings_equal compares buffer bindings", "[vkexec][pass]")
{
  using vkexec::detail::storage_bindings_equal;

  char buffer_left{};
  char buffer_right{};
  vkexec::storage_binding const left_binding{
    .buffer = fake_buffer(&buffer_left), .byte_size = k_byte_size, .binding = 0
  };
  vkexec::storage_binding const right_binding{
    .buffer = fake_buffer(&buffer_right),
    .byte_size = k_byte_size,
    .binding = 0,
  };
  vkexec::storage_binding const left_copy = left_binding;

  REQUIRE(storage_bindings_equal(std::span{ &left_binding, 1 }, std::span{ &left_copy, 1 }));
  REQUIRE_FALSE(storage_bindings_equal(std::span{ &left_binding, 1 }, std::span{ &right_binding, 1 }));
  REQUIRE(
    storage_bindings_equal(std::span<vkexec::storage_binding const>{}, std::span<vkexec::storage_binding const>{}));

  vkexec::storage_binding const diff_binding{
    .buffer = left_binding.buffer,
    .byte_size = left_binding.byte_size,
    .binding = 1,
  };
  vkexec::storage_binding const diff_size_binding{
    .buffer = left_binding.buffer,
    .byte_size = left_binding.byte_size + 1,
    .binding = left_binding.binding,
  };

  REQUIRE_FALSE(storage_bindings_equal(std::span{ &left_binding, 1 }, std::span{ &diff_size_binding, 1 }));
  REQUIRE_FALSE(storage_bindings_equal(std::span{ &left_binding, 1 }, std::span{ &diff_binding, 1 }));

  std::array const bindings{ left_binding, right_binding };
  std::array const bindings_reordered{ right_binding, left_binding };
  REQUIRE_FALSE(storage_bindings_equal(std::span{ bindings }, std::span{ bindings_reordered }));
  REQUIRE(storage_bindings_equal(std::span{ bindings }, std::span{ bindings }));
}

TEST_CASE("write_storage_descriptors returns when no buffers are bound", "[vkexec][pass]")
{ vkexec::write_storage_descriptors(VK_NULL_HANDLE, VK_NULL_HANDLE, std::span<vkexec::storage_binding const>{}); }
