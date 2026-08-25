#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vkexec/scheduler.hpp>
#include <vkexec/submit.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>

#include <stdexec/execution.hpp>

#include <concepts>
#include <type_traits>

namespace ex = stdexec;

namespace {

constexpr float k_clear_r = 0.1F;
constexpr float k_clear_g = 0.2F;
constexpr float k_clear_b = 0.3F;
constexpr float k_clear_a = 0.4F;

}// namespace

TEST_CASE("make_clear_values maps pipeline config to Vulkan clears", "[vkexec][graphics]")
{
  vkexec::graphics_pipeline_config cfg{};
  cfg.clear_r = k_clear_r;
  cfg.clear_g = k_clear_g;
  cfg.clear_b = k_clear_b;
  cfg.clear_a = k_clear_a;

  auto const clears = vkexec::make_clear_values(cfg);

  REQUIRE(clears.at(0).color.float32[0] == Catch::Approx(k_clear_r));
  REQUIRE(clears.at(0).color.float32[1] == Catch::Approx(k_clear_g));
  REQUIRE(clears.at(0).color.float32[2] == Catch::Approx(k_clear_b));
  REQUIRE(clears.at(0).color.float32[3] == Catch::Approx(k_clear_a));
  REQUIRE(clears.at(1).depthStencil.depth == Catch::Approx(vkexec::k_depth_clear_value));
  REQUIRE(clears.at(1).depthStencil.stencil == vkexec::k_stencil_clear_value);
}

TEST_CASE("make_clear_values uses default clear color from config", "[vkexec][graphics]")
{
  vkexec::graphics_pipeline_config const cfg{};
  auto const clears = vkexec::make_clear_values(cfg);

  REQUIRE(clears.at(0).color.float32[0] == Catch::Approx(vkexec::k_default_clear_r));
  REQUIRE(clears.at(0).color.float32[1] == Catch::Approx(vkexec::k_default_clear_g));
  REQUIRE(clears.at(0).color.float32[2] == Catch::Approx(vkexec::k_default_clear_b));
  REQUIRE(clears.at(0).color.float32[3] == Catch::Approx(vkexec::k_default_clear_a));
}

TEST_CASE("draw | submit yields stop-aware async sender", "[vkexec][graphics][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  vkexec::draw_sender const sync{
    .ctx = nullptr,
    .win = nullptr,
    .pipeline = nullptr,
    .vertex_count = 0,
  };
  auto const async_sender = sync | vkexec::submit;

  STATIC_REQUIRE(std::same_as<std::remove_cvref_t<decltype(async_sender)>, vkexec::draw_async_sender>);

  // NOLINTNEXTLINE(misc-include-cleaner)
  auto const completion = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(async_sender));
  REQUIRE(completion == sched);

  using signatures = vkexec::draw_async_sender::completion_signatures;
  STATIC_REQUIRE(std::same_as<signatures,
    ex::completion_signatures<ex::set_value_t(), ex::set_error_t(vkexec::error), ex::set_stopped_t()>>);
}
