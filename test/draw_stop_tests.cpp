#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/detail/sync_wait_outcome.hpp>
#include <vkexec/submit.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/triangle_shaders.hpp>
#include <vkexec_graphics/window.hpp>

#include <stdexec/execution.hpp>
#include <stdexec/stop_token.hpp>

#include <cstdint>
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
  auto outcome = vkexec::try_sync_wait(
    vkexec::window::headless({ .width = k_window_width, .height = k_window_height, .title = "vkexec draw stop tests" }));
  if (outcome.failed()) { vkexec::test::skip_if_no_vulkan(outcome.take_error()); }
  if (outcome.stopped || !outcome.values.has_value()) { FAIL("window::headless stopped unexpectedly"); }
  return vkexec::detail::take_sync_value(std::move(*outcome.values));
}

[[nodiscard]] auto make_triangle_pipeline(vkexec::window &win) -> vkexec::graphics_pipeline
{
  return vkexec::test::sync_wait_value(vkexec::graphics_pipeline::create(
    win.ctx(), win.render_pass(), vkexec::shaders::k_triangle_vert, vkexec::shaders::k_triangle_frag));
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
  auto const waited = vkexec::test::sync_wait_sender(std::move(env_sender));
  REQUIRE(vkexec::test::sync_wait_stopped(waited));
}

TEST_CASE("draw | submit reclaims frame slot when stop races with GPU completion", "[vkexec][draw][gpu]")
{
  // NOLINTNEXTLINE(misc-const-correctness) — draw() needs a mutable window reference
  headless_fixture fixture;
  ex::inplace_stop_source source;

  auto env_sender =
    ex::write_env(fixture.draw_submit_sender(), ex::prop{ ex::get_stop_token, source.get_token() });

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
