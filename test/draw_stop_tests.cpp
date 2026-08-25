#include <catch2/catch_test_macros.hpp>

#include <vkexec/config.hpp>
#include <vkexec/submit.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_edsl/types.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/window.hpp>

#include <stdexec/execution.hpp>
#include <stdexec/stop_token.hpp>

#include <cstdint>
#include <string>
#include <thread>
#include <utility>

namespace ex = stdexec;
namespace edsl = vkexec::edsl;

namespace {

constexpr std::uint32_t k_window_width = 64;
constexpr std::uint32_t k_window_height = 64;
constexpr std::uint32_t k_triangle_vertices = 3;
constexpr int k_frame_slots = 2;
constexpr int k_post_stop_frames = 4;

[[nodiscard]] auto make_headless_window() -> vkexec::window
{
  auto result = vkexec::window::headless(
    { .width = k_window_width, .height = k_window_height, .title = "vkexec draw stop tests" });
  if (!result) { SKIP(std::string("Headless surface unavailable: ") + std::string(result.error().message())); }
  return std::move(*result);
}

[[nodiscard]] auto make_triangle_pipeline(vkexec::window &win) -> vkexec::graphics_pipeline
{
  auto result = vkexec::graphics_pipeline::create(
    win.ctx(),
    win.render_pass(),
    [](edsl::Int vertex_id, edsl::VertexWriter out) -> void {
      edsl::Float2 const pos = edsl::select(vertex_id == edsl::Int::constant(0),
        edsl::vec2(0.0, -0.5),
        edsl::select(vertex_id == edsl::Int::constant(1), edsl::vec2(0.5, 0.5), edsl::vec2(-0.5, 0.5)));
      edsl::Float3 const col = edsl::select(vertex_id == edsl::Int::constant(0),
        edsl::vec3(1.0, 0.2, 0.2),
        edsl::select(vertex_id == edsl::Int::constant(1), edsl::vec3(0.2, 1.0, 0.2), edsl::vec3(0.2, 0.4, 1.0)));
      out.position(pos);
      out.color(col);
    },
    [](edsl::FragmentReader fragment_in, edsl::FragmentWriter out) -> void {
      out.color(edsl::vec4(fragment_in.color(), 1.0));
    });
  if (!result) {
    FAIL(std::string("graphics pipeline creation failed: ") + std::string(result.error().message()));
  }
  return std::move(*result);
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
    // NOLINTNEXTLINE(misc-include-cleaner)
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
    // NOLINTNEXTLINE(misc-include-cleaner)
    ex::write_env(fixture.draw_submit_sender(), ex::prop{ ex::get_stop_token, source.get_token() });

  std::jthread const stopper{ [&source]() -> void { source.request_stop(); } };

  // May complete with value or stopped depending on timing; borrowed-fence reclaim must not leak slots.
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
