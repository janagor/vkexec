#include <catch2/catch_test_macros.hpp>

#include <vkexec/buffer.hpp>
#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/sync_wait.hpp>

#include <stdexec/execution.hpp>
#include <stdexec/stop_token.hpp>
#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace ex = stdexec;

namespace {

constexpr std::size_t k_count = 8;
constexpr float k_fill = 2.5F;
constexpr std::uint32_t k_int_fill = 7U;

auto skip_if_no_vulkan(vkexec::error const &err) -> void
{ SKIP(std::string("Vulkan unavailable: ") + std::string(err.message())); }

}// namespace

TEST_CASE("buffer::allocate sender completes with a filled buffer", "[vkexec][buffer][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(ctx_result.error()); }
  auto &ctx = **ctx_result;

  auto waited = vkexec::sync_wait(vkexec::buffer<float>::allocate(ctx, k_count, k_fill));
  REQUIRE(waited.has_value());
  REQUIRE(waited->has_value());
  auto [values] = std::move(**waited);
  REQUIRE(values.size() == k_count);
  REQUIRE(values.vk_buffer() != VK_NULL_HANDLE);
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  REQUIRE(values.data()[0] == k_fill);
  REQUIRE(values.data()[k_count - 1] == k_fill);
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

TEST_CASE("buffer::create is an alias for allocate", "[vkexec][buffer][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(ctx_result.error()); }
  auto &ctx = **ctx_result;

  auto waited = vkexec::sync_wait(vkexec::buffer<std::uint32_t>::create(ctx, k_count, k_int_fill));
  REQUIRE(waited.has_value());
  REQUIRE(waited->has_value());
  auto [values] = std::move(**waited);
  REQUIRE(values.size() == k_count);
}

TEST_CASE("buffer::create_sync allocates synchronously", "[vkexec][buffer][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(ctx_result.error()); }
  auto &ctx = **ctx_result;

  auto values_result = vkexec::buffer<float>::create_sync(ctx, k_count, k_fill);
  REQUIRE(values_result.has_value());
  auto &values = *values_result;
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
  auto const waited =
    // NOLINTNEXTLINE(misc-include-cleaner)
    vkexec::sync_wait(ex::write_env(sender, ex::prop{ ex::get_stop_token, source.get_token() }));
  REQUIRE(waited.has_value());
  REQUIRE_FALSE(waited->has_value());
}

TEST_CASE("buffer allocate advertises completion scheduler", "[vkexec][buffer][scheduler]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(ctx_result.error()); }
  auto &ctx = **ctx_result;

  auto const sender = vkexec::buffer<float>::allocate(ctx, k_count);
  // NOLINTNEXTLINE(misc-include-cleaner)
  auto const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
  REQUIRE(sched == ctx.get_scheduler());
}
