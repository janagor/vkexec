#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/image.hpp>
#include <vkexec/image_view.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/rendering.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <utility>

namespace {

constexpr std::uint32_t k_width = 32;
constexpr std::uint32_t k_height = 32;

auto skip_if_unavailable(std::exception const &error) -> void
{ SKIP(std::string("Vulkan feature set unavailable: ") + error.what()); }

}// namespace

TEST_CASE("dynamic rendering begins and ends on a color target", "[vkexec][rendering][gpu]")
{
  VkPhysicalDeviceVulkan13Features features_13{};
  features_13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  features_13.dynamicRendering = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 3;
  requirements.require_extension_feature(features_13);

  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(vkexec::scheduler_options{ .requirements = std::move(requirements) }); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_unavailable(error); }

  auto img = vkexec::image::create(*ctx,
    vkexec::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::image_usage::color_storage,
    });
  auto view = vkexec::image_view::create(*ctx, img);

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
      .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      .dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .src_access = 0,
      .dst_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    });

  vkexec::color_attachment color{};
  color.view = view.handle();
  color.clear.color = { { 0.1F, 0.2F, 0.3F, 1.0F } };
  std::array<vkexec::color_attachment, 1> const colors{ color };

  vkexec::cmd_begin_rendering(cmd,
    vkexec::rendering_info{
      .extent = img.extent(),
      .color = colors,
    });
  vkexec::cmd_end_rendering(cmd);

  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
  std::array<VkCommandBuffer, 1> const cmds{ cmd };
  ctx->submit(vkexec::queue_submit{ .command_buffers = cmds });
  ctx->free_command_buffer(cmd);
}
