#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/image.hpp>
#include <vkexec/queue_submit.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <exception>
#include <optional>
#include <string>

namespace {

auto skip_if_no_vulkan(std::exception const &error) -> void
{ SKIP(std::string("Vulkan unavailable: ") + error.what()); }

}// namespace

TEST_CASE("image_barrier transitions a color image to general", "[vkexec][image][gpu]")
{
  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_no_vulkan(error); }

  auto img = vkexec::image::create(*ctx,
    vkexec::image_create_info{
      .width = 32,
      .height = 32,
      .usage = vkexec::image_usage::color_storage,
    });

  VkCommandBuffer cmd = ctx->allocate_command_buffer();
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
  ctx->submit(vkexec::queue_submit{ .command_buffers = cmds });
  ctx->free_command_buffer(cmd);
}
