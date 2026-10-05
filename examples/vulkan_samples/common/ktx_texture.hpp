#ifndef VKEXEC_EXAMPLES_KTX_TEXTURE_HPP
#define VKEXEC_EXAMPLES_KTX_TEXTURE_HPP

#include "sync_wait_helpers.hpp"

#include <vkexec/barrier.hpp>
#include <vkexec/context.hpp>
#include <vkexec/image_view.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec_vma/allocator.hpp>
#include <vkexec_vma/gpu_buffer.hpp>
#include <vkexec_vma/image.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <utility>
#include <vector>

namespace vkexec::examples {

struct ktx_texture
{
  vma::image image;
  owned::image_view view;
  vma::gpu_buffer staging;
  VkExtent2D extent{};
};

namespace detail {

  constexpr std::size_t k_ktx_header_bytes = 64;
  constexpr std::size_t k_ktx_format_offset = 28;
  constexpr std::size_t k_ktx_width_offset = 36;
  constexpr std::size_t k_ktx_height_offset = 40;
  constexpr std::size_t k_ktx_faces_offset = 52;
  constexpr std::size_t k_ktx_key_values_offset = 60;
  constexpr std::uint32_t k_astc_srgb_8x8_gl_format = 0x93D7;
  constexpr std::uint32_t k_astc_block_bytes = 16;
  constexpr std::uint32_t k_astc_8x8_block_extent = 8;
  constexpr std::uint32_t k_astc_5x5_block_extent = 5;
  constexpr std::size_t k_ktx2_level_index = 80;
  constexpr std::size_t k_ktx2_level_index_bytes = 24;
  constexpr std::size_t k_ktx2_format_offset = 12;
  constexpr std::size_t k_ktx2_width_offset = 20;
  constexpr std::size_t k_ktx2_height_offset = 24;
  constexpr std::size_t k_ktx2_faces_offset = 36;
  constexpr std::size_t k_ktx2_levels_offset = 40;
  constexpr std::size_t k_ktx2_supercompression_offset = 44;
  constexpr std::size_t k_ktx2_level_size_offset = sizeof(std::uint64_t);
  constexpr std::array<unsigned char, 12>
    k_ktx_identifier{ 0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A };
  constexpr std::array<unsigned char, 12>
    k_ktx2_identifier{ 0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A };

  [[nodiscard]] inline auto read_u32(std::span<char const> bytes, std::size_t offset) -> std::uint32_t
  {
    if (offset + sizeof(std::uint32_t) > bytes.size()) { fail_check("truncated KTX texture"); }
    std::uint32_t value = 0;
    std::memcpy(&value, bytes.subspan(offset, sizeof(value)).data(), sizeof(value));
    return value;
  }

  [[nodiscard]] inline auto read_u64(std::span<char const> bytes, std::size_t offset) -> std::uint64_t
  {
    if (offset + sizeof(std::uint64_t) > bytes.size()) { fail_check("truncated KTX texture"); }
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.subspan(offset, sizeof(value)).data(), sizeof(value));
    return value;
  }

}// namespace detail

[[nodiscard]] inline auto load_ktx_texture(context &ctx, vma::allocator &allocator, std::filesystem::path const &path)
  -> ktx_texture
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream) { fail_check("could not open KTX texture"); }
  std::vector<char> const file(std::istreambuf_iterator<char>{ stream }, std::istreambuf_iterator<char>{});
  std::span<char const> const bytes(file);
  if (bytes.size() < detail::k_ktx_header_bytes + sizeof(std::uint32_t)) { fail_check("truncated KTX texture"); }
  bool const ktx1 = std::memcmp(bytes.data(), detail::k_ktx_identifier.data(), detail::k_ktx_identifier.size()) == 0;
  bool const ktx2 = std::memcmp(bytes.data(), detail::k_ktx2_identifier.data(), detail::k_ktx2_identifier.size()) == 0;
  if (!ktx1 && !ktx2) { fail_check("unknown KTX texture version"); }
  VkExtent2D extent{};
  VkFormat format = VK_FORMAT_UNDEFINED;
  std::size_t payload_offset = 0;
  std::size_t image_size = 0;
  std::uint32_t block_extent = 0;
  if (ktx1) {
    if (detail::read_u32(bytes, detail::k_ktx_format_offset) != detail::k_astc_srgb_8x8_gl_format
        || detail::read_u32(bytes, detail::k_ktx_faces_offset) != 1) {
      fail_check("KTX1 texture must be a 2D ASTC 8x8 sRGB image");
    }
    extent = { .width = detail::read_u32(bytes, detail::k_ktx_width_offset),
      .height = detail::read_u32(bytes, detail::k_ktx_height_offset) };
    std::size_t const image_size_offset =
      detail::k_ktx_header_bytes + detail::read_u32(bytes, detail::k_ktx_key_values_offset);
    image_size = detail::read_u32(bytes, image_size_offset);
    payload_offset = image_size_offset + sizeof(std::uint32_t);
    format = VK_FORMAT_ASTC_8x8_SRGB_BLOCK;
    block_extent = detail::k_astc_8x8_block_extent;
  } else {
    if (bytes.size() < detail::k_ktx2_level_index + detail::k_ktx2_level_index_bytes
        || detail::read_u32(bytes, detail::k_ktx2_faces_offset) != 1
        || detail::read_u32(bytes, detail::k_ktx2_levels_offset) == 0
        || detail::read_u32(bytes, detail::k_ktx2_supercompression_offset) != 0) {
      fail_check("KTX2 texture must have an uncompressed 2D level");
    }
    extent = { .width = detail::read_u32(bytes, detail::k_ktx2_width_offset),
      .height = detail::read_u32(bytes, detail::k_ktx2_height_offset) };
    format = static_cast<VkFormat>(detail::read_u32(bytes, detail::k_ktx2_format_offset));
    if (format != VK_FORMAT_ASTC_5x5_SRGB_BLOCK) { fail_check("unsupported KTX2 texture format"); }
    block_extent = detail::k_astc_5x5_block_extent;
    std::uint64_t const level_offset = detail::read_u64(bytes, detail::k_ktx2_level_index);
    std::uint64_t const level_size =
      detail::read_u64(bytes, detail::k_ktx2_level_index + detail::k_ktx2_level_size_offset);
    if (!std::in_range<std::size_t>(level_offset) || !std::in_range<std::size_t>(level_size)) {
      fail_check("KTX2 level exceeds addressable size");
    }
    payload_offset = level_offset;
    image_size = level_size;
  }
  std::size_t const block_count = ((static_cast<std::size_t>(extent.width) + block_extent - 1) / block_extent)
                                  * ((static_cast<std::size_t>(extent.height) + block_extent - 1) / block_extent);
  if (extent.width == 0 || extent.height == 0 || image_size != block_count * detail::k_astc_block_bytes
      || payload_offset > bytes.size() || image_size > bytes.size() - payload_offset) {
    fail_check("invalid KTX texture payload");
  }

  auto staging = sync_wait_value(vma::factory::make_gpu_buffer(allocator,
    vma::gpu_buffer_create_info{
      .size = image_size,
      .memory = vma::buffer_memory::host_visible,
    }));
  std::memcpy(staging.mapped().data(), bytes.subspan(payload_offset, image_size).data(), image_size);
  if (auto flushed = staging.flush(); !flushed) { abort_with_error(flushed.error()); }

  auto image = sync_wait_value(allocate_image(allocator,
    image_create_info{
      .extent = { .width = extent.width, .height = extent.height, .depth = 1 },
      .format = format,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .queue_families = {},
    }));
  auto view = sync_wait_value(vma::factory::make_image_view(ctx, image));
  return ktx_texture{
    .image = std::move(image), .view = std::move(view), .staging = std::move(staging), .extent = extent
  };
}

inline auto record_ktx_texture_upload(context &ctx, VkCommandBuffer cmd, ktx_texture const &texture) -> void
{
  auto before = image_barrier(ctx,
    cmd,
    image_barrier_params{
      .image = texture.image.handle(),
      .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
      .new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .dst_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .dst_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
    });
  if (!before) { abort_with_error(before.error()); }
  VkBufferImageCopy copy{};
  copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy.imageSubresource.layerCount = 1;
  copy.imageExtent = { .width = texture.extent.width, .height = texture.extent.height, .depth = 1 };
  vkCmdCopyBufferToImage(
    cmd, texture.staging.handle(), texture.image.handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
  auto after = image_barrier(ctx,
    cmd,
    image_barrier_params{
      .image = texture.image.handle(),
      .old_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .src_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .dst_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dst_access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
    });
  if (!after) { abort_with_error(after.error()); }
}

}// namespace vkexec::examples

#endif
