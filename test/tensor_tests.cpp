#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/pipeline.hpp>
#include <vkexec/tensor.hpp>
#include <vkexec/tensor_sync.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ex = stdexec;

namespace {

constexpr std::size_t k_count = 8;
constexpr float k_fill = 1.5F;
constexpr float k_host_write = 3.25F;
constexpr std::uint32_t k_binding = 2;

}// namespace

TEST_CASE("tensor::create allocates staging-backed storage with a host mirror", "[vkexec][tensor][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto values = vkexec::test::sync_wait_value(vkexec::tensor<float>::create(*ctx, k_count, k_fill));
  REQUIRE(values.size() == k_count);
  REQUIRE(values.vk_buffer() != VK_NULL_HANDLE);
  REQUIRE(values.staging().handle() != VK_NULL_HANDLE);
  REQUIRE(values.device().handle() == values.vk_buffer());
  REQUIRE(values.byte_size() == k_count * sizeof(float));
  REQUIRE(values.span().front() == k_fill);
  REQUIRE(values.span().back() == k_fill);
}

TEST_CASE("tensor::create copies a span into the host mirror", "[vkexec][tensor][gpu]")
{
  auto ctx = vkexec::test::require_context();
  std::vector<std::uint32_t> const host{ 1, 2, 3, 4 };
  auto values = vkexec::test::sync_wait_value(vkexec::tensor<std::uint32_t>::create(*ctx, std::span{ host }));
  REQUIRE(values.size() == host.size());
  REQUIRE(values.span().front() == 1);
  REQUIRE(values.span().back() == 4);
}

TEST_CASE("tensor::storage_binding exposes the device buffer handle", "[vkexec][tensor][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto values = vkexec::test::sync_wait_value(vkexec::tensor<float>::create(*ctx, k_count, k_fill));
  vkexec::storage_binding const binding = values.storage_binding(k_binding);
  REQUIRE(binding.buffer == values.vk_buffer());
  REQUIRE(binding.byte_size == values.byte_size());
  REQUIRE(binding.binding == k_binding);
}

TEST_CASE("tensor upload/download round-trips host mirror through device storage", "[vkexec][tensor][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto values = vkexec::test::sync_wait_value(vkexec::tensor<float>::create(*ctx, k_count, k_fill));
  std::ranges::fill(values.span(), k_host_write);

  REQUIRE(values.upload(*ctx));
  std::ranges::fill(values.span(), 0.F);
  REQUIRE(values.download(*ctx));

  REQUIRE(values.span().front() == k_host_write);
  REQUIRE(values.span().back() == k_host_write);
}

TEST_CASE("sync_to_device/sync_to_host round-trip via pass graph", "[vkexec][tensor][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto values = vkexec::test::sync_wait_value(vkexec::tensor<float>::create(*ctx, k_count, k_fill));
  std::ranges::fill(values.span(), k_host_write);

  auto uploaded = vkexec::test::sync_wait_sender(
    ex::schedule(ctx->get_scheduler()) | vkexec::sync_to_device(values));
  REQUIRE(vkexec::test::sync_wait_completed(uploaded));
  std::ranges::fill(values.span(), -1.F);
  auto downloaded = vkexec::test::sync_wait_sender(
    ex::schedule(ctx->get_scheduler()) | vkexec::sync_to_host(values));
  REQUIRE(vkexec::test::sync_wait_completed(downloaded));

  REQUIRE(values.span().front() == k_host_write);
  REQUIRE(values.span().back() == k_host_write);
}
