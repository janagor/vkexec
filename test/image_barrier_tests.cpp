#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/result.hpp>
#include <vkexec/image.hpp>
#include <vkexec/queue_submit.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <memory>

namespace {

constexpr std::uint32_t k_width = 32;
constexpr std::uint32_t k_height = 32;

}// namespace

TEST_CASE("image_barrier transitions a color image to general", "[vkexec][image][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto img = vkexec::test::sync_wait_value(vkexec::image::create(*ctx,
    vkexec::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::image_usage::color_storage,
    }));

  auto cmd_result = ctx->allocate_command_buffer();
  REQUIRE(cmd_result.has_value());
  auto *cmd = vkexec::detail::expected_take(cmd_result);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);

  vkexec::image_barrier(cmd,
    {
      .image = img.handle(),
      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
      .new_layout = VK_IMAGE_LAYOUT_GENERAL,
      .src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      .dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      .src_access = 0,
      .dst_access = VK_ACCESS_SHADER_WRITE_BIT,
    });

  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
  std::array<VkCommandBuffer, 1> const cmds{ cmd };
  REQUIRE(ctx->submit(vkexec::queue_submit{ .command_buffers = cmds }));
  ctx->free_command_buffer(cmd);
}
