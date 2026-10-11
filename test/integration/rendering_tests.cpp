// NOLINTBEGIN(readability-function-cognitive-complexity)

#include "test_helpers.hpp"
#include "vma_test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/context.hpp>
#include <vkexec/image_view.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/result.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_extensions/dynamic_rendering/rendering.hpp>
#include <vkexec_features/dynamic_rendering.hpp>
#include <vkexec_features/feature.hpp>
#include <vkexec_vma/image.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
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

TEST_CASE("dynamic rendering begins and ends on a color target", "[vkexec][rendering][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 3;
  vkexec::feat::configure<vkexec::feat::dynamic_rendering>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto img = vkexec::test::sync_wait_value(vkexec::vma::factory::make_image(allocator,
    vkexec::vma::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::vma::image_usage::color_storage,
    }));
  auto view = vkexec::test::sync_wait_value(vkexec::vma::factory::make_image_view(*ctx, img));

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
      .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
      .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .src_stage = VK_PIPELINE_STAGE_2_NONE,
      .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      .src_access = 0,
      .dst_access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
    }));

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
  REQUIRE(ctx->submit_and_wait(cmd));
  ctx->free_command_buffer(cmd);
}

TEST_CASE("dynamic rendering to one mip preserves another mip in the graph", "[vkexec][rendering][gpu][ranges]")
{
  constexpr std::uint32_t k_api_major = 1;
  constexpr std::uint32_t k_api_minor = 3;
  constexpr std::uint32_t k_base_extent = 4;
  constexpr std::uint32_t k_target_extent = 2;
  constexpr std::uint32_t k_target_mip = 1;
  constexpr std::uint32_t k_mip_count = 2;
  constexpr std::uint32_t k_one = 1;
  constexpr VkDeviceSize k_pixel_bytes = 4;
  constexpr VkDeviceSize k_base_bytes =
    static_cast<VkDeviceSize>(k_base_extent) * static_cast<VkDeviceSize>(k_base_extent) * k_pixel_bytes;
  constexpr VkDeviceSize k_target_bytes =
    static_cast<VkDeviceSize>(k_target_extent) * static_cast<VkDeviceSize>(k_target_extent) * k_pixel_bytes;
  constexpr std::size_t k_channels = 4;
  constexpr std::byte k_full_channel{ 255 };
  constexpr std::array<std::byte, k_channels> k_base_color{ std::byte{}, k_full_channel, std::byte{}, k_full_channel };
  constexpr std::array<std::byte, k_channels> k_target_color{
    k_full_channel, std::byte{}, std::byte{}, k_full_channel
  };
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = k_api_major;
  requirements.api_version_minor = k_api_minor;
  vkexec::feat::configure<vkexec::feat::dynamic_rendering>(requirements);
  auto ctx = vkexec::test::require_context({ .requirements = std::move(requirements) });
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto image = vkexec::test::sync_wait_value(vkexec::allocate_image(allocator,
    vkexec::image_create_info{ .extent = { .width = k_base_extent, .height = k_base_extent, .depth = k_one },
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .usage = static_cast<VkImageUsageFlags>(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
               | static_cast<VkImageUsageFlags>(VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
               | static_cast<VkImageUsageFlags>(VK_IMAGE_USAGE_TRANSFER_DST_BIT),
      .mip_levels = k_mip_count,
      .queue_families = {} }));
  VkImageSubresourceRange const base{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = k_one,
    .baseArrayLayer = 0,
    .layerCount = k_one };
  auto target = base;
  target.baseMipLevel = k_target_mip;
  auto view = vkexec::test::sync_wait_value(vkexec::factory::make_image_view(
    *ctx, image.handle(), vkexec::image_view_create_info{ .format = VK_FORMAT_R8G8B8A8_UNORM, .range = target }));
  auto readback = vkexec::test::sync_wait_value(vkexec::allocate_buffer(allocator,
    vkexec::buffer_create_info{ .size = k_base_bytes + k_target_bytes,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .memory = vkexec::memory_domain::staging }));
  auto *const handle = image.handle();
  auto *const output = readback.handle();
  auto *const target_view = view.handle();
  auto whole = base;
  whole.levelCount = k_mip_count;
  auto graph = stdexec::schedule(ctx->get_scheduler())
               | vkexec::custom_pass(
                 vkexec::uses(vkexec::image_extent{ .image = handle, .mip_levels = k_mip_count, .array_layers = k_one },
                   vkexec::write(handle, base, vkexec::image_usage::transfer_destination)),
                 [handle, base](VkCommandBuffer cmd) -> void {
                   VkClearColorValue const color{ .float32 = { 0.0F, 1.0F, 0.0F, 1.0F } };
                   vkCmdClearColorImage(cmd, handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, k_one, &base);
                 })
               | vkexec::custom_pass(vkexec::uses(vkexec::write(handle, target, vkexec::image_usage::color_attachment)),
                 [target_view](VkCommandBuffer cmd) -> vkexec::status {
                   vkexec::color_attachment color{ .view = target_view };
                   color.clear.color = { { 1.0F, 0.0F, 0.0F, 1.0F } };
                   std::array const colors{ color };
                   VKEXEC_TRY(vkexec::cmd_begin_rendering(cmd,
                     vkexec::rendering_info{
                       .extent = { .width = k_target_extent, .height = k_target_extent }, .color = colors }));
                   return vkexec::cmd_end_rendering(cmd);
                 })
               | vkexec::custom_pass(vkexec::uses(vkexec::read(handle, whole, vkexec::image_usage::transfer_source),
                                       vkexec::write(output, vkexec::buffer_usage::transfer_destination)),
                 [handle, output](VkCommandBuffer cmd) -> void {
                   std::array<VkBufferImageCopy, k_mip_count> const copies{
                     VkBufferImageCopy{ .bufferOffset = 0,
                       .bufferRowLength = 0,
                       .bufferImageHeight = 0,
                       .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                         .mipLevel = 0,
                         .baseArrayLayer = 0,
                         .layerCount = k_one },
                       .imageOffset = {},
                       .imageExtent = { .width = k_base_extent, .height = k_base_extent, .depth = k_one } },
                     VkBufferImageCopy{ .bufferOffset = k_base_bytes,
                       .bufferRowLength = 0,
                       .bufferImageHeight = 0,
                       .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                         .mipLevel = k_target_mip,
                         .baseArrayLayer = 0,
                         .layerCount = k_one },
                       .imageOffset = {},
                       .imageExtent = { .width = k_target_extent, .height = k_target_extent, .depth = k_one } }
                   };
                   vkCmdCopyImageToBuffer(cmd,
                     handle,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     output,
                     static_cast<std::uint32_t>(copies.size()),
                     copies.data());
                 })
               | vkexec::barrier::buffer({ .buffer = output,
                 .src_stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                 .dst_stage = VK_PIPELINE_STAGE_2_HOST_BIT,
                 .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 .dst_access = VK_ACCESS_2_HOST_READ_BIT });
  auto const waited = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(waited));
  REQUIRE(readback.invalidate());
  auto const bytes = readback.mapped();
  for (std::size_t index{}; index < bytes.size(); ++index) {
    auto const &expected = index < k_base_bytes ? k_base_color : k_target_color;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    REQUIRE(bytes[index] == expected.at(index % k_channels));
  }
}

// NOLINTEND(readability-function-cognitive-complexity)
