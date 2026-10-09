#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>
#include <vkexec_graphics/present.hpp>
#include <vkexec_graphics/presenter.hpp>
#include <vkexec_graphics/submit.hpp>
#include <vkexec_graphics/triangle_shaders.hpp>
#include <vkexec_vma/graphics.hpp>

#include <stdexec/execution.hpp>
#include <stdexec/stop_token.hpp>
#include <vulkan/vulkan_core.h>

#include <array>
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
constexpr int k_repeated_aborts = 8;
constexpr std::uint32_t k_single_subresource = 1;

[[nodiscard]] auto make_headless_presenter() -> vkexec::owned::presenter
{
  auto outcome = vkexec::try_sync_wait_value(vkexec::factory::make_headless_presenter({
    .width = k_presenter_width,
    .height = k_presenter_height,
    .validation_layers = false,
    .surface_instance_extensions = {},
    .create_surface = {},
    .create_depth_attachment = vkexec::vma::make_depth_attachment_factory(),
    .requirements = {},
  }));
  if (!outcome) { vkexec::test::skip_if_no_vulkan(outcome.error()); }
  return vkexec::expected_take(outcome);
}

[[nodiscard]] auto make_triangle_pipeline(vkexec::owned::presenter &win) -> vkexec::owned::graphics_pipeline
{
  return vkexec::test::sync_wait_value(vkexec::factory::make_graphics_pipeline(
    win.ctx(), win.render_pass(), vkexec::shaders::k_triangle_vert, vkexec::shaders::k_triangle_frag));
}

struct headless_fixture
{
  vkexec::owned::presenter win;
  vkexec::owned::graphics_pipeline pipeline;

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

TEST_CASE("blocking draw observes pre-requested stop", "[vkexec][draw][gpu]")
{
  headless_fixture fixture;
  ex::inplace_stop_source source;
  source.request_stop();

  auto sender =
    ex::schedule(fixture.win.ctx().get_scheduler()) | vkexec::draw(fixture.win, fixture.pipeline, k_triangle_vertices);
  auto stopped_sender = ex::write_env(sender, ex::prop{ ex::get_stop_token, source.get_token() });
  auto const stopped = vkexec::test::sync_wait_sender(std::move(stopped_sender));
  REQUIRE(vkexec::test::sync_wait_stopped(stopped));

  auto retry =
    ex::schedule(fixture.win.ctx().get_scheduler()) | vkexec::draw(fixture.win, fixture.pipeline, k_triangle_vertices);
  auto const retried = vkexec::test::sync_wait_sender(retry);
  REQUIRE(vkexec::test::sync_wait_completed(retried));
}

TEST_CASE("draw | submit reclaims frame slot when stop races with GPU completion", "[vkexec][draw][gpu]")
{
  // NOLINTNEXTLINE(misc-const-correctness) — draw() needs a mutable presenter reference
  headless_fixture fixture;
  ex::inplace_stop_source source;

  auto env_sender = ex::write_env(fixture.draw_submit_sender(), ex::prop{ ex::get_stop_token, source.get_token() });

  // Concurrent stop: ensure the presenter can still acquire frames after a raced cancel.
  std::thread stopper{ [&source]() -> void { source.request_stop(); } };

  (void)vkexec::test::sync_wait_sender(std::move(env_sender));
  stopper.join();

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

TEST_CASE("graphics pass helpers leave command-buffer recording open", "[vkexec][draw][gpu]")
{
  headless_fixture fixture;
  auto begun = fixture.win.begin_frame();
  REQUIRE(begun.has_value());
  REQUIRE(begun->has_value());
  vkexec::frame const frame = begun->value_or(vkexec::frame{});

  auto allocated = fixture.win.ctx().allocate_command_buffer(fixture.win.ctx().graphics_queue_ref());
  REQUIRE(allocated.has_value());
  VkCommandBuffer cmd = vkexec::expected_take(allocated);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
  vkexec::record_draw_pass(cmd,
    fixture.win.render_pass(),
    frame.framebuffer,
    frame.extent,
    fixture.pipeline.config(),
    fixture.pipeline.bind(),
    k_triangle_vertices);
  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);

  REQUIRE(vkResetCommandBuffer(cmd, 0) == VK_SUCCESS);
  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
  fixture.pipeline.record_pass(cmd, fixture.win.render_pass(), frame.framebuffer, frame.extent, k_triangle_vertices);
  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);

  fixture.win.ctx().free_command_buffer(cmd);
  REQUIRE(fixture.win.end_frame(frame));
  fixture.win.wait_idle();
}

TEST_CASE("presenter finishes once before external submission", "[vkexec][draw][gpu]")
{
  headless_fixture fixture;
  auto begun = fixture.win.begin_frame();
  REQUIRE(begun.has_value());
  REQUIRE(begun->has_value());
  vkexec::frame const frame = begun->value_or(vkexec::frame{});
  fixture.pipeline.record_pass(
    frame.command_buffer, fixture.win.render_pass(), frame.framebuffer, frame.extent, k_triangle_vertices);

  REQUIRE(fixture.win.finish_frame_recording(frame));
  REQUIRE_FALSE(fixture.win.finish_frame_recording(frame));
  auto sync = fixture.win.submission_sync(frame);
  REQUIRE(sync.has_value());

  std::array<VkCommandBuffer, 1> const commands{ frame.command_buffer };
  std::array<vkexec::semaphore_submit, 1> const waits{ sync->image_available_wait };
  std::array<vkexec::semaphore_submit, 1> const signals{ sync->render_finished_signal };
  REQUIRE(fixture.win.ctx().submit(vkexec::queue_submit{
    .command_buffers = commands,
    .waits = waits,
    .signals = signals,
    .fence = sync->fence,
    .queue = fixture.win.ctx().graphics_queue(),
  }));
  REQUIRE(fixture.win.present_submitted(frame));
  fixture.win.wait_idle();
}

TEST_CASE("presenter end_frame accepts an explicitly finished frame", "[vkexec][draw][gpu]")
{
  headless_fixture fixture;
  auto begun = fixture.win.begin_frame();
  REQUIRE(begun.has_value());
  REQUIRE(begun->has_value());
  vkexec::frame const frame = begun->value_or(vkexec::frame{});
  fixture.pipeline.record_pass(
    frame.command_buffer, fixture.win.render_pass(), frame.framebuffer, frame.extent, k_triangle_vertices);

  REQUIRE(fixture.win.finish_frame_recording(frame));
  REQUIRE(fixture.win.end_frame(frame));
  fixture.win.wait_idle();
}

TEST_CASE("presenter rejects presenting a recording frame", "[vkexec][draw][gpu]")
{
  headless_fixture fixture;
  auto begun = fixture.win.begin_frame();
  REQUIRE(begun.has_value());
  REQUIRE(begun->has_value());
  vkexec::frame const frame = begun->value_or(vkexec::frame{});
  fixture.pipeline.record_pass(
    frame.command_buffer, fixture.win.render_pass(), frame.framebuffer, frame.extent, k_triangle_vertices);

  REQUIRE_FALSE(fixture.win.present_submitted(frame));
  REQUIRE(fixture.win.end_frame(frame));
  fixture.win.wait_idle();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("cancelled presentation graph releases its acquired frame", "[vkexec][draw][gpu]")
{
  auto win = make_headless_presenter();
  ex::inplace_stop_source source;
  source.request_stop();
  for (int index{}; index < k_repeated_aborts; ++index) {
    auto acquired = win.acquire_frame();
    REQUIRE(acquired.has_value());
    REQUIRE(acquired->has_value());
    vkexec::acquired_frame const frame = acquired->value_or(vkexec::acquired_frame{});
    auto graph = ex::schedule(win.ctx().get_scheduler()) | vkexec::on_queue(win.ctx().graphics_queue_ref())
                 | vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {})
                 | vkexec::present(win, frame);
    auto stopped = ex::write_env(std::move(graph), ex::prop{ ex::get_stop_token, source.get_token() });
    auto const outcome = vkexec::test::sync_wait_sender(std::move(stopped));
    REQUIRE(vkexec::test::sync_wait_stopped(outcome));
  }

  auto retry = win.acquire_frame();
  REQUIRE(retry.has_value());
  REQUIRE(retry->has_value());
  vkexec::acquired_frame const retry_frame = retry->value_or(vkexec::acquired_frame{});
  REQUIRE(win.abandon_frame(retry_frame));
}

TEST_CASE("invalid presentation graph releases its acquired frame", "[vkexec][draw][gpu]")
{
  auto win = make_headless_presenter();
  auto acquired = win.acquire_frame();
  REQUIRE(acquired.has_value());
  REQUIRE(acquired->has_value());
  vkexec::acquired_frame const frame = acquired->value_or(vkexec::acquired_frame{});

  auto graph = ex::schedule(win.ctx().get_scheduler()) | vkexec::on_queue(vkexec::queue_ref{})
               | vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {})
               | vkexec::present(win, frame);
  auto const outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(outcome.failed());

  auto retry = win.acquire_frame();
  REQUIRE(retry.has_value());
  REQUIRE(retry->has_value());
  vkexec::acquired_frame const retry_frame = retry->value_or(vkexec::acquired_frame{});
  REQUIRE(win.abandon_frame(retry_frame));
}

TEST_CASE("recording failure releases an acquired presentation frame", "[vkexec][draw][gpu]")
{
  auto win = make_headless_presenter();
  auto acquired = win.acquire_frame();
  REQUIRE(acquired.has_value());
  REQUIRE(acquired->has_value());
  vkexec::acquired_frame const frame = acquired->value_or(vkexec::acquired_frame{});

  VkImageSubresourceRange const color_range{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = k_single_subresource,
    .baseArrayLayer = 0,
    .layerCount = k_single_subresource };
  auto color_use = vkexec::write(frame.image, color_range, vkexec::image_usage::color_attachment);
  color_use.final_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  auto graph = ex::schedule(win.ctx().get_scheduler()) | vkexec::on_queue(win.ctx().graphics_queue_ref())
               | vkexec::custom_pass(vkexec::uses(color_use),
                 [](VkCommandBuffer /*cmd*/) -> vkexec::status {
                   return vkexec::fail(vkexec::errc::invalid_argument, "intentional recording failure");
                 })
               | vkexec::present(win, frame);
  auto const outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(outcome.failed());

  auto retry = win.acquire_frame();
  REQUIRE(retry.has_value());
  REQUIRE(retry->has_value());
  vkexec::acquired_frame const retry_frame = retry->value_or(vkexec::acquired_frame{});
  REQUIRE(win.abandon_frame(retry_frame));
}

TEST_CASE("abandoning a frame restores a reset presentation fence", "[vkexec][draw][gpu]")
{
  auto win = make_headless_presenter();
  auto acquired = win.acquire_frame();
  REQUIRE(acquired.has_value());
  REQUIRE(acquired->has_value());
  vkexec::acquired_frame const frame = acquired->value_or(vkexec::acquired_frame{});
  auto sync = win.submission_sync(frame);
  REQUIRE(sync.has_value());
  VkFence fence = sync->fence;
  REQUIRE(vkResetFences(win.ctx().device(), k_single_subresource, &fence) == VK_SUCCESS);
  REQUIRE(win.abandon_frame(frame));

  auto retry = win.acquire_frame();
  REQUIRE(retry.has_value());
  REQUIRE(retry->has_value());
  vkexec::acquired_frame const retry_frame = retry->value_or(vkexec::acquired_frame{});
  REQUIRE(win.abandon_frame(retry_frame));
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("presentation graph submits and presents an acquired frame", "[vkexec][draw][gpu]")
{
  headless_fixture fixture;
  auto acquired = fixture.win.acquire_frame();
  REQUIRE(acquired.has_value());
  REQUIRE(acquired->has_value());
  vkexec::acquired_frame const frame = acquired->value_or(vkexec::acquired_frame{});
  VkImageSubresourceRange const color_range{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = k_single_subresource,
    .baseArrayLayer = 0,
    .layerCount = k_single_subresource };
  auto color_use = vkexec::write(frame.image, color_range, vkexec::image_usage::color_attachment);
  color_use.final_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  auto sync = fixture.win.submission_sync(frame);
  REQUIRE(sync.has_value());
  VkFence fence = sync->fence;

  auto graph = ex::schedule(fixture.win.ctx().get_scheduler())
               | vkexec::on_queue(fixture.win.ctx().graphics_queue_ref())
               | vkexec::custom_pass(vkexec::uses(color_use),
                 [&](VkCommandBuffer cmd) -> void {
                   fixture.pipeline.record_pass(
                     cmd, fixture.win.render_pass(), frame.framebuffer, frame.extent, k_triangle_vertices);
                 })
               | vkexec::present(fixture.win, frame);
  auto const outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
  REQUIRE(vkGetFenceStatus(fixture.win.ctx().device(), fence) == VK_SUCCESS);

  auto retry = fixture.win.acquire_frame();
  REQUIRE(retry.has_value());
  REQUIRE(retry->has_value());
  vkexec::acquired_frame const retry_frame = retry->value_or(vkexec::acquired_frame{});
  REQUIRE(fixture.win.abandon_frame(retry_frame));
}

TEST_CASE("typed branches join before the final presentation pass", "[vkexec][draw][gpu]")
{
  headless_fixture fixture;
  auto acquired = fixture.win.acquire_frame();
  REQUIRE(acquired.has_value());
  REQUIRE(acquired->has_value());
  vkexec::acquired_frame const frame = acquired->value_or(vkexec::acquired_frame{});
  VkImageSubresourceRange const color_range{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = k_single_subresource,
    .baseArrayLayer = 0,
    .layerCount = k_single_subresource };
  auto color_use = vkexec::write(frame.image, color_range, vkexec::image_usage::color_attachment);
  color_use.final_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  auto sync = fixture.win.submission_sync(frame);
  REQUIRE(sync.has_value());
  VkFence fence = sync->fence;

  auto make_empty_pass = []() -> decltype(auto) {
    return vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {});
  };
  auto const graphics = fixture.win.ctx().graphics_queue_ref();
  auto const compute = fixture.win.ctx().compute_queue_ref();
  auto graph =
    ex::schedule(fixture.win.ctx().get_scheduler()) | vkexec::on_queue(graphics) | make_empty_pass()
    | vkexec::when_all(vkexec::on_queue(compute) | make_empty_pass(), vkexec::on_queue(graphics) | make_empty_pass())
    | vkexec::on_queue(graphics)
    | vkexec::custom_pass(vkexec::uses(color_use),
      [&](VkCommandBuffer cmd) -> void {
        fixture.pipeline.record_pass(
          cmd, fixture.win.render_pass(), frame.framebuffer, frame.extent, k_triangle_vertices);
      })
    | vkexec::present(fixture.win, frame);

  auto const outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
  REQUIRE(vkGetFenceStatus(fixture.win.ctx().device(), fence) == VK_SUCCESS);
}

TEST_CASE("draw_layers presents with presenter-owned recording", "[vkexec][draw][gpu]")
{
  headless_fixture fixture;
  auto const waited = vkexec::test::sync_wait_sender(
    ex::schedule(fixture.win.ctx().get_scheduler())
    | vkexec::draw_layers(fixture.win, { { .pipeline = &fixture.pipeline, .vertex_count = k_triangle_vertices } }));
  REQUIRE(vkexec::test::sync_wait_completed(waited));
  fixture.win.wait_idle();
}
