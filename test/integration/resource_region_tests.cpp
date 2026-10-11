// NOLINTBEGIN(readability-function-cognitive-complexity)

#include "test_helpers.hpp"
#include "vma_test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/resource_use.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

namespace {

constexpr VkDeviceSize k_buffer_size = 512;
constexpr VkDeviceSize k_half_size = 256;
constexpr VkDeviceSize k_snapshot_begin = 128;
constexpr VkDeviceSize k_overwrite_begin = 192;
constexpr VkDeviceSize k_overwrite_size = 128;
constexpr std::uint32_t k_first_pattern = 0x12345678U;
constexpr std::uint32_t k_second_pattern = 0xABCDEF01U;
constexpr std::uint32_t k_overwrite_pattern = 0x87654321U;
constexpr std::uint32_t k_copy_count = 1;
constexpr std::uint32_t k_mip_count = 4;
constexpr std::uint32_t k_image_extent = 8;
constexpr std::uint32_t k_one_layer = 1;
constexpr std::size_t k_channels = 4;
constexpr std::byte k_full_channel{ 255 };
constexpr std::array<std::byte, k_channels> k_expected_pixel{ k_full_channel,
  std::byte{},
  k_full_channel,
  k_full_channel };

[[nodiscard]] constexpr auto mip_range(std::uint32_t mip, std::uint32_t count = 1) -> VkImageSubresourceRange
{
  return { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = mip,
    .levelCount = count,
    .baseArrayLayer = 0,
    .layerCount = k_one_layer };
}

[[nodiscard]] auto host_read_barrier(VkBuffer buffer) -> decltype(auto)
{
  return vkexec::barrier::buffer({ .buffer = buffer,
    .src_stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
    .dst_stage = VK_PIPELINE_STAGE_2_HOST_BIT,
    .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
    .dst_access = VK_ACCESS_2_HOST_READ_BIT });
}

}// namespace

TEST_CASE("buffer suballocations preserve data across partial graph overlaps", "[vkexec][gpu][ranges]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto source = vkexec::test::sync_wait_value(vkexec::allocate_buffer(allocator,
    vkexec::buffer_create_info{ .size = k_buffer_size,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .memory = vkexec::memory_domain::device_local }));
  auto readback = vkexec::test::sync_wait_value(vkexec::allocate_buffer(allocator,
    vkexec::buffer_create_info{ .size = k_half_size + k_buffer_size,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .memory = vkexec::memory_domain::staging }));
  auto *const buffer = source.handle();
  auto *const output = readback.handle();
  auto graph =
    stdexec::schedule(ctx->get_scheduler())
    | vkexec::custom_pass(
      vkexec::uses(vkexec::buffer_extent{ .buffer = buffer, .size = k_buffer_size },
        vkexec::buffer_extent{ .buffer = output, .size = k_half_size + k_buffer_size },
        vkexec::write(buffer, 0, k_half_size, vkexec::buffer_usage::transfer_destination),
        vkexec::write(buffer, k_half_size, k_half_size, vkexec::buffer_usage::transfer_destination)),
      [buffer](VkCommandBuffer cmd) -> void {
        vkCmdFillBuffer(cmd, buffer, 0, k_half_size, k_first_pattern);
        vkCmdFillBuffer(cmd, buffer, k_half_size, k_half_size, k_second_pattern);
      })
    | vkexec::custom_pass(
      vkexec::uses(vkexec::read(buffer, k_snapshot_begin, k_half_size, vkexec::buffer_usage::transfer_source),
        vkexec::write(output, 0, k_half_size, vkexec::buffer_usage::transfer_destination)),
      [buffer, output](VkCommandBuffer cmd) -> void {
        VkBufferCopy const copy{ .srcOffset = k_snapshot_begin, .dstOffset = 0, .size = k_half_size };
        vkCmdCopyBuffer(cmd, buffer, output, k_copy_count, &copy);
      })
    | vkexec::custom_pass(vkexec::uses(vkexec::write(
                            buffer, k_overwrite_begin, k_overwrite_size, vkexec::buffer_usage::transfer_destination)),
      [buffer](VkCommandBuffer cmd) -> void {
        vkCmdFillBuffer(cmd, buffer, k_overwrite_begin, k_overwrite_size, k_overwrite_pattern);
      })
    | vkexec::custom_pass(
      vkexec::uses(vkexec::read(buffer, vkexec::buffer_usage::transfer_source),
        vkexec::write(output, k_half_size, k_buffer_size, vkexec::buffer_usage::transfer_destination)),
      [buffer, output](VkCommandBuffer cmd) -> void {
        VkBufferCopy const copy{ .srcOffset = 0, .dstOffset = k_half_size, .size = k_buffer_size };
        vkCmdCopyBuffer(cmd, buffer, output, k_copy_count, &copy);
      })
    | host_read_barrier(output);
  auto const waited = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(waited));
  REQUIRE(readback.invalidate());
  auto const bytes = readback.mapped();
  for (VkDeviceSize offset{}; offset < readback.size(); offset += sizeof(std::uint32_t)) {
    std::uint32_t actual{};
    auto const word = bytes.subspan(static_cast<std::size_t>(offset), sizeof(actual));
    std::memcpy(&actual, word.data(), sizeof(actual));
    bool const snapshot = offset < k_half_size;
    auto const source_offset = snapshot ? offset + k_snapshot_begin : offset - k_half_size;
    auto const expected = [snapshot, source_offset]() -> std::uint32_t {
      if (!snapshot && source_offset >= k_overwrite_begin && source_offset < k_overwrite_begin + k_overwrite_size) {
        return k_overwrite_pattern;
      }
      if (source_offset < k_half_size) { return k_first_pattern; }
      return k_second_pattern;
    }();
    REQUIRE(actual == expected);
  }
}

TEST_CASE("mipmap graph uses disjoint mips in each blit and verifies every output", "[vkexec][gpu][ranges]")
{
  auto ctx = vkexec::test::require_context();
  constexpr VkFormat k_format = VK_FORMAT_R8G8B8A8_UNORM;
  VkFormatProperties properties{};
  vkGetPhysicalDeviceFormatProperties(ctx->physical_device(), k_format, &properties);
  constexpr VkFormatFeatureFlags k_required_features = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
  if ((properties.optimalTilingFeatures & k_required_features) != k_required_features) {
    SKIP("RGBA8 blits unavailable");
  }
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto image = vkexec::test::sync_wait_value(vkexec::allocate_image(allocator,
    vkexec::image_create_info{ .extent = { .width = k_image_extent, .height = k_image_extent, .depth = 1 },
      .format = k_format,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .mip_levels = k_mip_count,
      .queue_families = {} }));
  std::array<VkBufferImageCopy, k_mip_count> copies{};
  VkDeviceSize readback_size{};
  for (std::uint32_t mip{}; mip < k_mip_count; ++mip) {
    auto const extent = k_image_extent >> mip;
    copies.at(mip) = VkBufferImageCopy{ .bufferOffset = readback_size,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel = mip,
        .baseArrayLayer = 0,
        .layerCount = k_one_layer },
      .imageOffset = {},
      .imageExtent = { .width = extent, .height = extent, .depth = 1 } };
    readback_size += static_cast<VkDeviceSize>(extent) * extent * k_channels;
  }
  auto readback = vkexec::test::sync_wait_value(vkexec::allocate_buffer(allocator,
    vkexec::buffer_create_info{
      .size = readback_size, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .memory = vkexec::memory_domain::staging }));
  auto *const handle = image.handle();
  auto *const output = readback.handle();
  auto const blit = [handle](std::uint32_t source_mip) -> decltype(auto) {
    auto const destination_mip = source_mip + 1;
    return vkexec::custom_pass(
      vkexec::uses(vkexec::read(handle, mip_range(source_mip), vkexec::image_usage::transfer_source),
        vkexec::write(handle, mip_range(destination_mip), vkexec::image_usage::transfer_destination)),
      [handle, source_mip, destination_mip](VkCommandBuffer cmd) -> void {
        auto const source_extent = static_cast<std::int32_t>(k_image_extent >> source_mip);
        auto const destination_extent = static_cast<std::int32_t>(k_image_extent >> destination_mip);
        VkImageBlit const region{ .srcSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                    .mipLevel = source_mip,
                                    .baseArrayLayer = 0,
                                    .layerCount = k_one_layer },
          .srcOffsets = { {}, { .x = source_extent, .y = source_extent, .z = 1 } },
          .dstSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = destination_mip,
            .baseArrayLayer = 0,
            .layerCount = k_one_layer },
          .dstOffsets = { {}, { .x = destination_extent, .y = destination_extent, .z = 1 } } };
        vkCmdBlitImage(cmd,
          handle,
          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          handle,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          k_copy_count,
          &region,
          VK_FILTER_NEAREST);
      });
  };
  constexpr std::uint32_t k_first_mip = 0;
  constexpr std::uint32_t k_second_mip = 1;
  constexpr std::uint32_t k_third_mip = 2;
  auto graph =
    stdexec::schedule(ctx->get_scheduler())
    | vkexec::custom_pass(
      vkexec::uses(vkexec::image_extent{ .image = handle, .mip_levels = k_mip_count, .array_layers = k_one_layer },
        vkexec::write(handle, mip_range(k_first_mip), vkexec::image_usage::transfer_destination)),
      [handle](VkCommandBuffer cmd) -> void {
        VkClearColorValue const color{ .float32 = { 1.0F, 0.0F, 1.0F, 1.0F } };
        auto const first = mip_range(k_first_mip);
        vkCmdClearColorImage(cmd, handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, k_copy_count, &first);
      })
    | blit(k_first_mip) | blit(k_second_mip) | blit(k_third_mip)
    | vkexec::custom_pass(
      vkexec::uses(vkexec::read(handle, mip_range(k_first_mip, k_mip_count), vkexec::image_usage::transfer_source),
        vkexec::write(output, vkexec::buffer_usage::transfer_destination)),
      [handle, output, copies](VkCommandBuffer cmd) -> void {
        vkCmdCopyImageToBuffer(cmd,
          handle,
          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          output,
          static_cast<std::uint32_t>(copies.size()),
          copies.data());
      })
    | host_read_barrier(output);
  auto const waited = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(waited));
  REQUIRE(readback.invalidate());
  auto const bytes = readback.mapped();
  for (std::size_t index{}; index < bytes.size(); ++index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    REQUIRE(bytes[index] == k_expected_pixel.at(index % k_channels));
  }
}

// NOLINTEND(readability-function-cognitive-complexity)
