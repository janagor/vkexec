#include <catch2/catch_test_macros.hpp>

#include <vkexec/error.hpp>
#include <vkexec/submit.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/triangle_shaders.hpp>
#include <vkexec_graphics/window.hpp>

#include <stdexec/execution.hpp>
#include <stdexec/stop_token.hpp>

#include <cstdint>
#include <string>
#include <thread>
#include <utility>

namespace ex = stdexec;

namespace {

constexpr std::uint32_t k_window_width = 64;
constexpr std::uint32_t k_window_height = 64;
constexpr std::uint32_t k_triangle_vertices = 3;
constexpr int k_frame_slots = 2;
constexpr int k_post_stop_frames = 4;

[[nodiscard]] auto make_headless_window() -> vkexec::window
{
  auto result =
    vkexec::window::headless({ .width = k_window_width, .height = k_window_height, .title = "vkexec draw stop tests" });
  if (!result) {
    SKIP(std::string("Headless surface unavailable: ") + std::string(vkexec::to_error(result.error()).message()));
  }
  return vkexec::detail::leaf_take(result);
}

[[nodiscard]] auto make_triangle_pipeline(vkexec::window &win) -> vkexec::graphics_pipeline
{
  auto result = vkexec::graphics_pipeline::create(
    win.ctx(), win.render_pass(), vkexec::shaders::k_triangle_vert, vkexec::shaders::k_triangle_frag);
  if (!result) {
    FAIL(std::string("graphics pipeline creation failed: ") + std::string(vkexec::to_error(result.error()).message()));
  }
  return vkexec::detail::leaf_take(result);
}

struct headless_fixture
{
  vkexec::window win;
  vkexec::graphics_pipeline pipeline;

  headless_fixture() : win(make_headless_window()), pipeline(make_triangle_pipeline(win)) {}

  [[nodiscard]] auto draw_submit_sender() -> auto
  {
    return ex::schedule(win.ctx().get_scheduler()) | vkexec::draw(win, pipeline, k_triangle_vertices) | vkexec::submit;
  }
};

}// namespace

TEST_CASE("draw | submit completes with set_stopped when stop is already requested", "[vkexec][draw][gpu]")
{
  // NOLINTNEXTLINE(misc-const-correctness) — draw() needs a mutable window reference
  headless_fixture fixture;
  ex::inplace_stop_source source;
  source.request_stop();

  auto env_sender =
    ex::write_env(fixture.draw_submit_sender(), ex::prop{ ex::get_stop_token, source.get_token() });
  auto const waited = vkexec::sync_wait(std::move(env_sender));
  REQUIRE(waited.has_value());
  REQUIRE_FALSE(waited->has_value());
}

TEST_CASE("draw | submit reclaims frame slot when stop races with GPU completion", "[vkexec][draw][gpu]")
{
  // NOLINTNEXTLINE(misc-const-correctness) — draw() needs a mutable window reference
  headless_fixture fixture;
  ex::inplace_stop_source source;

  auto env_sender =
    ex::write_env(fixture.draw_submit_sender(), ex::prop{ ex::get_stop_token, source.get_token() });

  std::jthread const stopper{ [&source]() -> void { source.request_stop(); } };

  (void)vkexec::sync_wait(std::move(env_sender));

  for (int frame = 0; frame < k_post_stop_frames; ++frame) {
    auto const retry = vkexec::sync_wait(fixture.draw_submit_sender());
    REQUIRE(retry.has_value());
    REQUIRE(retry->has_value());
  }
}

TEST_CASE("draw | submit presents multiple headless frames without leaking frame slots", "[vkexec][draw][gpu]")
{
  headless_fixture fixture;

  for (int frame = 0; frame < k_frame_slots + k_post_stop_frames; ++frame) {
    auto const waited = vkexec::sync_wait(fixture.draw_submit_sender());
    REQUIRE(waited.has_value());
    REQUIRE(waited->has_value());
  }

  fixture.win.wait_idle();
}
