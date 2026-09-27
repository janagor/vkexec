#include "test_helpers.hpp"
#include "vma_test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/context.hpp>
#include <vkexec/result.hpp>
#include <vkexec_vma/image.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>

namespace {

constexpr std::uint32_t k_width = 32;
constexpr std::uint32_t k_height = 32;

}// namespace

TEST_CASE("image_barrier transitions a color image to general", "[vkexec][image][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto img = vkexec::test::sync_wait_value(vkexec::vma::factory::make_image(allocator,
    vkexec::vma::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::vma::image_usage::color_storage,
    }));

  auto cmd_result = ctx->allocate_command_buffer();
  REQUIRE(cmd_result.has_value());
  auto *cmd = vkexec::expected_take(cmd_result);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);

  REQUIRE(vkexec::image_barrier(*ctx,
    cmd,
    {
      .image = img.handle(),
      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
      .new_layout = VK_IMAGE_LAYOUT_GENERAL,
      .src_stage = VK_PIPELINE_STAGE_2_NONE,
      .dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .src_access = 0,
      .dst_access = VK_ACCESS_2_SHADER_WRITE_BIT,
    }));

  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
  REQUIRE(ctx->submit_and_wait(cmd));
  ctx->free_command_buffer(cmd);
}
