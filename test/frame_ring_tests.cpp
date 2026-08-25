#include <catch2/catch_test_macros.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/frame_ring.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/vulkan_requirements.hpp>

namespace {

auto skip_if_unavailable(std::exception const &error) -> void
{ SKIP(std::string("Vulkan feature set unavailable: ") + error.what()); }

auto open_timeline_context(std::optional<vkexec::context> &ctx) -> void
{
  VkPhysicalDeviceVulkan12Features features_12{};
  features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features_12.timelineSemaphore = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 2;
  requirements.require_extension_feature(features_12);

  VKEXEC_TRY { ctx.emplace(vkexec::scheduler_options{ .requirements = std::move(requirements) }); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_unavailable(error); }
}

auto record_empty(vkexec::context &ctx) -> VkCommandBuffer
{
  VkCommandBuffer cmd = ctx.allocate_command_buffer();
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
  return cmd;
}

}// namespace

TEST_CASE("frame_ring creates slot and image semaphores", "[vkexec][frame_ring][gpu]")
{
  std::optional<vkexec::context> ctx;
  open_timeline_context(ctx);

  auto ring = vkexec::frame_ring::create(*ctx,
    vkexec::frame_ring::create_info{ .slot_count = 2, .image_count = 3 });
  REQUIRE(ring.slot_count() == 2);
  REQUIRE(ring.image_count() == 3);
  REQUIRE(ring.acquire_semaphore(0) != VK_NULL_HANDLE);
  REQUIRE(ring.acquire_semaphore(1) != VK_NULL_HANDLE);
  REQUIRE(ring.render_finished_semaphore(2) != VK_NULL_HANDLE);
  REQUIRE(ring.timeline().handle() != VK_NULL_HANDLE);

  ring.wait_slot(0);
  ring.wait_image(0);
}

TEST_CASE("frame_ring gates slot reuse via timeline", "[vkexec][frame_ring][gpu]")
{
  std::optional<vkexec::context> ctx;
  open_timeline_context(ctx);

  auto ring = vkexec::frame_ring::create(*ctx,
    vkexec::frame_ring::create_info{ .slot_count = 2, .image_count = 2 });

  // Prime the acquire semaphore so the wait is satisfied without a real swapchain acquire.
  {
    std::array<VkSemaphore, 1> const signals{ ring.acquire_semaphore(0) };
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = signals.data();
    REQUIRE(vkQueueSubmit(ctx->graphics_queue() != VK_NULL_HANDLE ? ctx->graphics_queue() : ctx->compute_queue(),
              1,
              &submit,
              VK_NULL_HANDLE)
            == VK_SUCCESS);
  }

  auto const signal_value = ring.allocate_signal_value();
  auto const sync = ring.make_submit_sync(0, 0, signal_value);
  std::array<VkCommandBuffer, 1> const cmds{ record_empty(*ctx) };

  ctx->submit(vkexec::queue_submit{
    .command_buffers = cmds,
    .waits = sync.waits,
    .signals = sync.signals,
  });
  ring.mark_submitted(0, 0, signal_value);

  ring.wait_slot(0);
  ring.wait_image(0);
}

TEST_CASE("frame_ring resize_images replaces finished semaphores", "[vkexec][frame_ring][gpu]")
{
  std::optional<vkexec::context> ctx;
  open_timeline_context(ctx);

  auto ring = vkexec::frame_ring::create(*ctx,
    vkexec::frame_ring::create_info{ .slot_count = 2, .image_count = 2 });
  VkSemaphore const old_finished = ring.render_finished_semaphore(0);

  ring.resize_images(4);
  REQUIRE(ring.image_count() == 4);
  REQUIRE(ring.render_finished_semaphore(0) != VK_NULL_HANDLE);
  REQUIRE(ring.render_finished_semaphore(0) != old_finished);
  REQUIRE(ring.render_finished_semaphore(3) != VK_NULL_HANDLE);
  ring.reset_completion_tracking();
  ring.wait_slot(1);
  ring.wait_image(3);
}
