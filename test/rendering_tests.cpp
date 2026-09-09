#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/context.hpp>
#include <vkexec/result.hpp>
#include <vkexec/image.hpp>
#include <vkexec/image_view.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec_extensions/dynamic_rendering/rendering.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_features/dynamic_rendering.hpp>
#include <vkexec_features/feature.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

namespace {

constexpr std::uint32_t k_width = 32;
constexpr std::uint32_t k_height = 32;
constexpr float k_clear_r = 0.1F;
constexpr float k_clear_g = 0.2F;
constexpr float k_clear_b = 0.3F;
constexpr float k_clear_a = 1.0F;

}// namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("dynamic rendering begins and ends on a color target", "[vkexec][rendering][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 3;
  vkexec::feat::configure<vkexec::feat::dynamic_rendering>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::context::create({ .requirements = std::move(requirements) }));
  auto img = vkexec::test::sync_wait_value(vkexec::image::create(*ctx,
    vkexec::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::image_usage::color_storage,
    }));
  auto view = vkexec::test::sync_wait_value(vkexec::image_view::create(*ctx, img));

  auto cmd_result = ctx->allocate_command_buffer();
  REQUIRE(cmd_result.has_value());
  auto *cmd = vkexec::expected_take(cmd_result);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);

  vkexec::image_barrier(cmd,
    {
      .image = img.handle(),
      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
      .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      .dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .src_access = 0,
      .dst_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    });

  vkexec::color_attachment color{};
  color.view = view.handle();
  color.clear.color = { { k_clear_r, k_clear_g, k_clear_b, k_clear_a } };
  std::array<vkexec::color_attachment, 1> const colors{ color };

  REQUIRE(vkexec::cmd_begin_rendering(cmd,
    vkexec::rendering_info{
      .extent = img.extent(),
      .color = colors,
    }));
  REQUIRE(vkexec::cmd_end_rendering(cmd));

  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
  std::array<VkCommandBuffer, 1> const cmds{ cmd };
  REQUIRE(ctx->submit(vkexec::queue_submit{ .command_buffers = cmds }));
  ctx->free_command_buffer(cmd);
}
