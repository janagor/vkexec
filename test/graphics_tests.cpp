#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vkexec_graphics/graphics.hpp>

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
