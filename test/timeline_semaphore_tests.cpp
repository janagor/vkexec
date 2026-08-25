#include <catch2/catch_test_macros.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <utility>

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/timeline_semaphore.hpp>
#include <vkexec/vulkan_requirements.hpp>

namespace {

auto skip_if_unavailable(std::exception const &error) -> void
{ SKIP(std::string("Vulkan feature set unavailable: ") + error.what()); }

}// namespace

TEST_CASE("timeline_semaphore create and wait for initial value", "[vkexec][timeline][gpu]")
{
  VkPhysicalDeviceVulkan12Features features_12{};
  features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features_12.timelineSemaphore = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 2;
  requirements.require_extension_feature(features_12);

  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(vkexec::scheduler_options{ .requirements = std::move(requirements) }); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_unavailable(error); }

  auto timeline = vkexec::timeline_semaphore::create(*ctx, 3);
  REQUIRE(timeline.handle() != VK_NULL_HANDLE);
  timeline.wait(3);
  timeline.wait(0);
}

TEST_CASE("context::submit signals a timeline semaphore", "[vkexec][timeline][gpu]")
{
  VkPhysicalDeviceVulkan12Features features_12{};
  features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features_12.timelineSemaphore = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 2;
  requirements.require_extension_feature(features_12);

  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(vkexec::scheduler_options{ .requirements = std::move(requirements) }); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_unavailable(error); }

  auto timeline = vkexec::timeline_semaphore::create(*ctx, 0);
  VkCommandBuffer cmd = ctx->allocate_command_buffer();
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);

  constexpr std::uint64_t k_signal_value = 7;
  std::array<VkCommandBuffer, 1> const cmds{ cmd };
  std::array<vkexec::semaphore_submit, 1> const signals{ vkexec::semaphore_submit{
    .semaphore = timeline.handle(),
    .value = k_signal_value,
  } };

  ctx->submit(vkexec::queue_submit{
    .command_buffers = cmds,
    .signals = signals,
  });
  timeline.wait(k_signal_value);
  ctx->free_command_buffer(cmd);
}
