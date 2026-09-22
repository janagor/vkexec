#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/result.hpp>
#include <vkexec/submit.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>
#include <vkexec_graphics/presenter.hpp>
#include <vkexec_graphics/triangle_shaders.hpp>

#include <stdexec/execution.hpp>
#include <stdexec/stop_token.hpp>
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <thread>
#include <utility>

namespace ex = stdexec;

namespace {

constexpr std::uint32_t k_presenter_width = 64;
constexpr std::uint32_t k_presenter_height = 64;
constexpr std::uint32_t k_triangle_vertices = 3;
constexpr int k_frame_slots = 2;
constexpr int k_post_stop_frames = 4;

[[nodiscard]] auto make_headless_presenter() -> vkexec::presenter
{
  auto outcome = vkexec::try_sync_wait_value(vkexec::factory::make_headless_presenter({
    .width = k_presenter_width,
    .height = k_presenter_height,
    .validation_layers = false,
    .surface_instance_extensions = {},
    .create_surface = {},
    .requirements = {},
  }));
  if (!outcome) { vkexec::test::skip_if_no_vulkan(outcome.error()); }
  return vkexec::expected_take(outcome);
}

[[nodiscard]] auto make_triangle_pipeline(vkexec::presenter &win) -> vkexec::graphics_pipeline
{
  return vkexec::test::sync_wait_value(vkexec::factory::make_graphics_pipeline(
    win.ctx(), win.render_pass(), vkexec::shaders::k_triangle_vert, vkexec::shaders::k_triangle_frag));
}

struct headless_fixture
{
  vkexec::presenter win;
  vkexec::graphics_pipeline pipeline;

  headless_fixture() : win(make_headless_presenter()), pipeline(make_triangle_pipeline(win)) {}

  [[nodiscard]] auto draw_submit_sender() -> auto
  {
    return ex::schedule(win.ctx().get_scheduler()) | vkexec::draw(win, pipeline, k_triangle_vertices) | vkexec::submit;
  }
};

}// namespace

TEST_CASE("draw | submit completes with set_stopped when stop is already requested", "[vkexec][draw][gpu]")
{
  // NOLINTNEXTLINE(misc-const-correctness) — draw() needs a mutable presenter reference
  headless_fixture fixture;
  ex::inplace_stop_source source;
  source.request_stop();

  auto env_sender = ex::write_env(fixture.draw_submit_sender(), ex::prop{ ex::get_stop_token, source.get_token() });
  auto const waited = vkexec::test::sync_wait_sender(std::move(env_sender));
  REQUIRE(vkexec::test::sync_wait_stopped(waited));
}

TEST_CASE("draw | submit reclaims frame slot when stop races with GPU completion", "[vkexec][draw][gpu]")
{
  // NOLINTNEXTLINE(misc-const-correctness) — draw() needs a mutable presenter reference
  headless_fixture fixture;
  ex::inplace_stop_source source;

  auto env_sender = ex::write_env(fixture.draw_submit_sender(), ex::prop{ ex::get_stop_token, source.get_token() });

  // Concurrent stop: ensure the presenter can still acquire frames after a raced cancel.
  std::jthread const stopper{ [&source]() -> void { source.request_stop(); } };

  (void)vkexec::test::sync_wait_sender(std::move(env_sender));

  for (int frame = 0; frame < k_post_stop_frames; ++frame) {
    auto const retry = vkexec::test::sync_wait_sender(fixture.draw_submit_sender());
    REQUIRE(vkexec::test::sync_wait_completed(retry));
  }
}

TEST_CASE("draw | submit presents multiple headless frames without leaking frame slots", "[vkexec][draw][gpu]")
{
  headless_fixture fixture;

  for (int frame = 0; frame < k_frame_slots + k_post_stop_frames; ++frame) {
    auto const waited = vkexec::test::sync_wait_sender(fixture.draw_submit_sender());
    REQUIRE(vkexec::test::sync_wait_completed(waited));
  }

  fixture.win.wait_idle();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("presenter suspends at zero extent and resumes after resize", "[vkexec][draw][gpu]")
{
  auto win = make_headless_presenter();

  REQUIRE(win.resize(0, 0));
  REQUIRE(win.needs_resize());
  auto suspended = win.begin_frame();
  REQUIRE(suspended.has_value());
  REQUIRE_FALSE(suspended->has_value());

  REQUIRE(win.resize(k_presenter_width, k_presenter_height));
  REQUIRE_FALSE(win.needs_resize());
  auto resumed = win.begin_frame();
  REQUIRE(resumed.has_value());
  if (!resumed->has_value()) { FAIL("presenter remained suspended after non-zero resize"); }
  vkexec::frame const resumed_frame = **resumed;
  REQUIRE(vkEndCommandBuffer(resumed_frame.command_buffer) == VK_SUCCESS);
  REQUIRE(win.end_frame(resumed_frame));
  win.wait_idle();
}

TEST_CASE("borrowable graphics resources draw without owning pipeline", "[vkexec][draw][gpu][execution]")
{
  auto win = make_headless_presenter();
  auto resources_result = vkexec::create(win.ctx(),
    win.render_pass(),
    vkexec::graphics_pipeline_config{},
    vkexec::shaders::k_triangle_vert,
    vkexec::shaders::k_triangle_frag);
  REQUIRE(resources_result.has_value());
  auto resources = vkexec::expected_take(resources_result);

  auto const waited = vkexec::test::sync_wait_sender(ex::schedule(win.ctx().get_scheduler())
                                                     | vkexec::draw(win, resources, VK_NULL_HANDLE, k_triangle_vertices)
                                                     | vkexec::submit);
  REQUIRE(vkexec::test::sync_wait_completed(waited));

  win.wait_idle();
  vkexec::destroy(win.ctx(), resources);
}
