#include <catch2/catch_test_macros.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/frame_ring.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/vulkan_requirements.hpp>

namespace {

auto skip_if_unavailable(vkexec::error const &err) -> void
{ SKIP(std::string("Vulkan feature set unavailable: ") + std::string(err.message())); }

auto open_timeline_context() -> std::unique_ptr<vkexec::context>
{
  VkPhysicalDeviceVulkan12Features features_12{};
  features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features_12.timelineSemaphore = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 2;
  requirements.require_extension_feature(features_12);

  auto ctx_result = vkexec::context::create({ .requirements = std::move(requirements) });
  if (!ctx_result) { skip_if_unavailable(vkexec::to_error(ctx_result.error())); }
  return std::move(*ctx_result);
}

auto record_empty(vkexec::context &ctx) -> VkCommandBuffer
{
  auto cmd_result = ctx.allocate_command_buffer();
  REQUIRE(cmd_result.has_value());
  VkCommandBuffer cmd = *cmd_result;
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
  auto ctx = open_timeline_context();

  auto ring_result =
    vkexec::frame_ring::create(*ctx, vkexec::frame_ring::create_info{ .slot_count = 2, .image_count = 3 });
  REQUIRE(ring_result.has_value());
  auto &ring = *ring_result;
  REQUIRE(ring.slot_count() == 2);
  REQUIRE(ring.image_count() == 3);
  REQUIRE(ring.acquire_semaphore(0).value_or(VK_NULL_HANDLE) != VK_NULL_HANDLE);
  REQUIRE(ring.acquire_semaphore(1).value_or(VK_NULL_HANDLE) != VK_NULL_HANDLE);
  REQUIRE(ring.render_finished_semaphore(2).value_or(VK_NULL_HANDLE) != VK_NULL_HANDLE);
  REQUIRE(ring.timeline().handle() != VK_NULL_HANDLE);

  REQUIRE(ring.wait_slot(0).has_value());
  REQUIRE(ring.wait_image(0).has_value());
}

TEST_CASE("frame_ring gates slot reuse via timeline", "[vkexec][frame_ring][gpu]")
{
  auto ctx = open_timeline_context();

  auto ring_result =
    vkexec::frame_ring::create(*ctx, vkexec::frame_ring::create_info{ .slot_count = 2, .image_count = 2 });
  REQUIRE(ring_result.has_value());
  auto &ring = *ring_result;

  // Prime the acquire semaphore so the wait is satisfied without a real swapchain acquire.
  {
    auto const acquire = ring.acquire_semaphore(0);
    REQUIRE(acquire.has_value());
    std::array<VkSemaphore, 1> const signals{ *acquire };
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
  REQUIRE(sync.has_value());
  std::array<VkCommandBuffer, 1> const cmds{ record_empty(*ctx) };

  REQUIRE(ctx
      ->submit(vkexec::queue_submit{
        .command_buffers = cmds,
        .waits = sync->waits,
        .signals = sync->signals,
      })
      .has_value());
  REQUIRE(ring.mark_submitted(0, 0, signal_value).has_value());

  REQUIRE(ring.wait_slot(0).has_value());
  REQUIRE(ring.wait_image(0).has_value());
}

TEST_CASE("frame_ring resize_images replaces finished semaphores", "[vkexec][frame_ring][gpu]")
{
  auto ctx = open_timeline_context();

  auto ring_result =
    vkexec::frame_ring::create(*ctx, vkexec::frame_ring::create_info{ .slot_count = 2, .image_count = 2 });
  REQUIRE(ring_result.has_value());
  auto &ring = *ring_result;
  auto const old_finished = ring.render_finished_semaphore(0);
  REQUIRE(old_finished.has_value());

  REQUIRE(ring.resize_images(4).has_value());
  REQUIRE(ring.image_count() == 4);
  auto const new_finished = ring.render_finished_semaphore(0);
  REQUIRE(new_finished.has_value());
  REQUIRE(*new_finished != VK_NULL_HANDLE);
  REQUIRE(*new_finished != *old_finished);
  REQUIRE(ring.render_finished_semaphore(3).value_or(VK_NULL_HANDLE) != VK_NULL_HANDLE);
  ring.reset_completion_tracking();
  REQUIRE(ring.wait_slot(1).has_value());
  REQUIRE(ring.wait_image(3).has_value());
}
