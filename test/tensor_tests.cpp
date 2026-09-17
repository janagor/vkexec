#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/pipeline.hpp>
#include <vkexec/tensor.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

constexpr std::size_t k_count = 8;
constexpr float k_fill = 1.5F;
constexpr std::uint32_t k_binding = 2;

}// namespace

TEST_CASE("tensor::create allocates a filled host-visible buffer", "[vkexec][tensor][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto values = vkexec::test::sync_wait_value(vkexec::tensor<float>::create(*ctx, k_count, k_fill));
  REQUIRE(values.size() == k_count);
  REQUIRE(values.vk_buffer() != VK_NULL_HANDLE);
  REQUIRE(values.byte_size() == k_count * sizeof(float));
  REQUIRE(values.span().front() == k_fill);
  REQUIRE(values.span().back() == k_fill);
}

TEST_CASE("tensor::create copies a span into storage", "[vkexec][tensor][gpu]")
{
  auto ctx = vkexec::test::require_context();
  std::vector<std::uint32_t> const host{ 1, 2, 3, 4 };
  auto values = vkexec::test::sync_wait_value(vkexec::tensor<std::uint32_t>::create(*ctx, std::span{ host }));
  REQUIRE(values.size() == host.size());
  REQUIRE(values.span().front() == 1);
  REQUIRE(values.span().back() == 4);
}

TEST_CASE("tensor::storage_binding exposes buffer handle and binding index", "[vkexec][tensor][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto values = vkexec::test::sync_wait_value(vkexec::tensor<float>::create(*ctx, k_count, k_fill));
  vkexec::storage_binding const binding = values.storage_binding(k_binding);
  REQUIRE(binding.buffer == values.vk_buffer());
  REQUIRE(binding.byte_size == values.byte_size());
  REQUIRE(binding.binding == k_binding);
}
