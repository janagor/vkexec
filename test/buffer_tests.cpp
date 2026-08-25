#include <catch2/catch_test_macros.hpp>

#include <vkexec/buffer.hpp>
#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/scheduler.hpp>

#include <stdexec/execution.hpp>
#include <stdexec/stop_token.hpp>
#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <utility>

namespace ex = stdexec;

namespace {

constexpr std::size_t k_count = 8;
constexpr float k_fill = 2.5F;
constexpr std::uint32_t k_int_fill = 7U;

auto skip_if_no_vulkan(std::exception const &error) -> void
{ SKIP(std::string("Vulkan unavailable: ") + error.what()); }

}// namespace

TEST_CASE("buffer::allocate sender completes with a filled buffer", "[vkexec][buffer][gpu]")
{
  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_no_vulkan(error); }

  auto result = ex::sync_wait(vkexec::buffer<float>::allocate(*ctx, k_count, k_fill));
  REQUIRE(result.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  auto [values] = std::move(result).value();
  REQUIRE(values.size() == k_count);
  REQUIRE(values.vk_buffer() != VK_NULL_HANDLE);
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  REQUIRE(values.data()[0] == k_fill);
  REQUIRE(values.data()[k_count - 1] == k_fill);
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

TEST_CASE("buffer::create is an alias for allocate", "[vkexec][buffer][gpu]")
{
  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_no_vulkan(error); }

  auto result = ex::sync_wait(vkexec::buffer<std::uint32_t>::create(*ctx, k_count, k_int_fill));
  REQUIRE(result.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  auto [values] = std::move(result).value();
  REQUIRE(values.size() == k_count);
}

TEST_CASE("sync buffer ctor is sync_wait of allocate", "[vkexec][buffer][gpu]")
{
  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_no_vulkan(error); }

  vkexec::buffer<float> values(*ctx, k_count, k_fill);
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
  auto const result =
    // NOLINTNEXTLINE(misc-include-cleaner)
    ex::sync_wait(ex::write_env(sender, ex::prop{ ex::get_stop_token, source.get_token() }));
  REQUIRE_FALSE(result.has_value());
}

TEST_CASE("buffer allocate advertises completion scheduler", "[vkexec][buffer][scheduler]")
{
  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_no_vulkan(error); }

  auto const sender = vkexec::buffer<float>::allocate(*ctx, k_count);
  // NOLINTNEXTLINE(misc-include-cleaner)
  auto const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
  REQUIRE(sched == ctx->get_scheduler());
}
