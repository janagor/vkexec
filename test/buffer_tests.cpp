#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <stdexec/__detail/__execution_fwd.hpp>
#include <vkexec/buffer.hpp>
#include <vkexec/context.hpp>
#include <vkexec/scheduler.hpp>

#include <stdexec/execution.hpp>
#include <stdexec/stop_token.hpp>
#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ex = stdexec;

namespace {

constexpr std::size_t k_count = 8;
constexpr float k_fill = 2.5F;
constexpr std::uint32_t k_int_fill = 7U;

}// namespace

TEST_CASE("buffer::allocate sender completes with a filled buffer", "[vkexec][buffer][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto values = vkexec::test::sync_wait_value(vkexec::buffer<float>::allocate(*ctx, k_count, k_fill));
  REQUIRE(values.size() == k_count);
  REQUIRE(values.vk_buffer() != VK_NULL_HANDLE);
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  REQUIRE(values.data()[0] == k_fill);
  REQUIRE(values.data()[k_count - 1] == k_fill);
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

TEST_CASE("buffer::create is an alias for allocate", "[vkexec][buffer][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto values = vkexec::test::sync_wait_value(vkexec::buffer<std::uint32_t>::create(*ctx, k_count, k_int_fill));
  REQUIRE(values.size() == k_count);
}

TEST_CASE("sync_wait_value completes buffer::allocate", "[vkexec][buffer][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto values = vkexec::test::sync_wait_value(vkexec::buffer<float>::allocate(*ctx, k_count, k_fill));
  REQUIRE(values.size() == k_count);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  REQUIRE(values.data()[0] == k_fill);
}

TEST_CASE("buffer::allocate completes with set_stopped when stop is already requested", "[vkexec][buffer]")
{
  vkexec::context *const no_ctx = nullptr;
  ex::inplace_stop_source source;
  source.request_stop();

  auto sender = vkexec::buffer_allocate_sender<float>{ .ctx = no_ctx, .count = k_count, .fill = k_fill };
  auto const waited = vkexec::test::sync_wait_sender(
    ex::write_env(sender, ex::prop{ ex::get_stop_token, source.get_token() }));
  REQUIRE(vkexec::test::sync_wait_stopped(waited));
}

TEST_CASE("buffer allocate advertises completion scheduler", "[vkexec][buffer][scheduler]")
{
  auto ctx = vkexec::test::require_context();

  auto const sender = vkexec::buffer<float>::allocate(*ctx, k_count);
  auto const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
  REQUIRE(sched == ctx->get_scheduler());
}
