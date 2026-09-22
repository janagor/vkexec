#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include <vkexec/context.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_extensions/timeline_semaphore/timeline_semaphore.hpp>
#include <vkexec_features/feature.hpp>
#include <vkexec_features/timeline_semaphore.hpp>

namespace {

constexpr std::uint64_t k_signal_value = 7;

}// namespace

TEST_CASE("timeline_semaphore create and wait for initial value", "[vkexec][timeline][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 2;
  vkexec::feat::configure<vkexec::feat::timeline_semaphore>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
  auto timeline = vkexec::test::sync_wait_value(vkexec::factory::make_timeline_semaphore(*ctx, 3));
  REQUIRE(timeline.handle() != VK_NULL_HANDLE);
  REQUIRE(timeline.wait(3));
  REQUIRE(timeline.wait(0));
}

TEST_CASE("context::submit signals a timeline semaphore", "[vkexec][timeline][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 2;
  vkexec::feat::configure<vkexec::feat::timeline_semaphore>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
  auto timeline = vkexec::test::sync_wait_value(vkexec::factory::make_timeline_semaphore(*ctx, 0));

  auto cmd_result = ctx->allocate_command_buffer();
  REQUIRE(cmd_result.has_value());
  auto *cmd = vkexec::expected_take(cmd_result);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);

  std::array<VkCommandBuffer, 1> const cmds{ cmd };
  std::array<vkexec::semaphore_submit, 1> const signals{ vkexec::semaphore_submit{
    .semaphore = timeline.handle(),
    .value = k_signal_value,
  } };

  REQUIRE(ctx->submit(vkexec::queue_submit{
    .command_buffers = cmds,
    .signals = signals,
  }));
  REQUIRE(timeline.wait(k_signal_value));
  ctx->free_command_buffer(cmd);
}
