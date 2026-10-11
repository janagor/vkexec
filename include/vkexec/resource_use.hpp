#ifndef VKEXEC_RESOURCE_USE_HPP
#define VKEXEC_RESOURCE_USE_HPP

//! \file
//! Resource declarations for custom pass recording.
//!
//! Buffer byte ranges and image aspect/mip/layer ranges are tracked independently.
//! Dependencies and barriers cover only intersecting tracked regions. Within one
//! pass, disjoint uses and compatible overlapping reads without a new layout or
//! ownership transition are allowed. Combine conflicting overlaps in one use.
//! Combined depth/stencil masks remain coupled and cannot be mixed with separate
//! depth or stencil declarations for the same image. Plane aspects are unsupported.
//! Register extents before first use when finite and symbolic ranges can meet at
//! the actual resource end, so possibly empty symbolic tails are eliminated.

#include <vulkan/vulkan.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace vkexec {

enum class resource_access : std::uint8_t { read, write, read_write };

enum class image_usage : std::uint8_t {
  sampled_compute,
  sampled_fragment,
  sampled_graphics,
  storage_compute,
  storage_graphics,
  color_attachment,
  depth_attachment,
  transfer_source,
  transfer_destination,
};

enum class buffer_usage : std::uint8_t {
  indirect,
  vertex,
  index,
  uniform_compute,
  uniform_vertex,
  uniform_fragment,
  storage_compute,
  storage_graphics,
  transfer_source,
  transfer_destination,
};

struct image_use
{
  VkImage image{ VK_NULL_HANDLE };
  //! Barrier subresources, not 3D slice-view coordinates. VK_REMAINING_* is supported.
  //! Separate depth/stencil aspects require a suitable format and enabled Vulkan features.
  VkImageSubresourceRange range{};
  image_usage usage{};
  resource_access access{};
  //! Applies only to previously untracked subresources; required for their first read.
  VkImageLayout initial_layout{ VK_IMAGE_LAYOUT_UNDEFINED };
  //! Overrides the usage's default layout when a descriptor requires another valid layout.
  VkImageLayout layout{ VK_IMAGE_LAYOUT_UNDEFINED };
  //! Layout after this use, for render-pass finalLayout transitions.
  VkImageLayout final_layout{ VK_IMAGE_LAYOUT_UNDEFINED };
};

struct buffer_use
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  VkDeviceSize offset{};
  VkDeviceSize size{ VK_WHOLE_SIZE };
  buffer_usage usage{};
  resource_access access{};
};

//! Optional bounds and actual sharing mode registered before the first graph use of this buffer.
struct buffer_extent
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  VkDeviceSize size{};
  VkSharingMode sharing_mode{ VK_SHARING_MODE_EXCLUSIVE };
};

//! Optional bounds and actual sharing mode registered before the first graph use of this image.
struct image_extent
{
  VkImage image{ VK_NULL_HANDLE };
  std::uint32_t mip_levels{};
  std::uint32_t array_layers{};
  VkSharingMode sharing_mode{ VK_SHARING_MODE_EXCLUSIVE };
};

[[nodiscard]] constexpr auto read(VkImage image, VkImageSubresourceRange range, image_usage usage) noexcept -> image_use
{ return { .image = image, .range = range, .usage = usage, .access = resource_access::read }; }

//! Declares the known layout of an image before its first graph access.
[[nodiscard]] constexpr auto
  read(VkImage image, VkImageSubresourceRange range, image_usage usage, VkImageLayout initial_layout) noexcept
  -> image_use
{
  return {
    .image = image, .range = range, .usage = usage, .access = resource_access::read, .initial_layout = initial_layout
  };
}

[[nodiscard]] constexpr auto write(VkImage image, VkImageSubresourceRange range, image_usage usage) noexcept
  -> image_use
{ return { .image = image, .range = range, .usage = usage, .access = resource_access::write }; }

[[nodiscard]] constexpr auto
  write(VkImage image, VkImageSubresourceRange range, image_usage usage, VkImageLayout initial_layout) noexcept
  -> image_use
{
  return {
    .image = image, .range = range, .usage = usage, .access = resource_access::write, .initial_layout = initial_layout
  };
}

[[nodiscard]] constexpr auto read_write(VkImage image, VkImageSubresourceRange range, image_usage usage) noexcept
  -> image_use
{ return { .image = image, .range = range, .usage = usage, .access = resource_access::read_write }; }

[[nodiscard]] constexpr auto
  read_write(VkImage image, VkImageSubresourceRange range, image_usage usage, VkImageLayout initial_layout) noexcept
  -> image_use
{
  return { .image = image,
    .range = range,
    .usage = usage,
    .access = resource_access::read_write,
    .initial_layout = initial_layout };
}

[[nodiscard]] constexpr auto read(VkBuffer buffer, buffer_usage usage) noexcept -> buffer_use
{ return { .buffer = buffer, .usage = usage, .access = resource_access::read }; }

[[nodiscard]] constexpr auto read(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, buffer_usage usage) noexcept
  -> buffer_use
{ return { .buffer = buffer, .offset = offset, .size = size, .usage = usage, .access = resource_access::read }; }

[[nodiscard]] constexpr auto write(VkBuffer buffer, buffer_usage usage) noexcept -> buffer_use
{ return { .buffer = buffer, .usage = usage, .access = resource_access::write }; }

[[nodiscard]] constexpr auto write(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, buffer_usage usage) noexcept
  -> buffer_use
{ return { .buffer = buffer, .offset = offset, .size = size, .usage = usage, .access = resource_access::write }; }

[[nodiscard]] constexpr auto read_write(VkBuffer buffer, buffer_usage usage) noexcept -> buffer_use
{ return { .buffer = buffer, .usage = usage, .access = resource_access::read_write }; }

[[nodiscard]] constexpr auto
  read_write(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, buffer_usage usage) noexcept -> buffer_use
{ return { .buffer = buffer, .offset = offset, .size = size, .usage = usage, .access = resource_access::read_write }; }

template<class T>
concept resource_use = std::same_as<T, image_use> || std::same_as<T, buffer_use>;

template<class T>
concept resource_declaration = resource_use<T> || std::same_as<T, buffer_extent> || std::same_as<T, image_extent>;

template<resource_declaration... Uses> struct resource_uses
{
  std::array<image_use, (std::size_t{} + ... + std::size_t{ std::same_as<Uses, image_use> })> images{};
  std::array<buffer_use, (std::size_t{} + ... + std::size_t{ std::same_as<Uses, buffer_use> })> buffers{};
  std::array<image_extent, (std::size_t{} + ... + std::size_t{ std::same_as<Uses, image_extent> })> image_extents{};
  std::array<buffer_extent, (std::size_t{} + ... + std::size_t{ std::same_as<Uses, buffer_extent> })> buffer_extents{};
};

template<resource_declaration... Uses> [[nodiscard]] constexpr auto uses(Uses... declarations) -> resource_uses<Uses...>
{
  resource_uses<Uses...> result{};
  std::size_t image_index{};
  std::size_t buffer_index{};
  std::size_t image_extent_index{};
  std::size_t buffer_extent_index{};
  [[maybe_unused]] auto append = [&]<resource_declaration Use>(Use declaration) constexpr -> void {
    if constexpr (std::same_as<Use, image_use>) {
      result.images.at(image_index++) = declaration;
    } else if constexpr (std::same_as<Use, buffer_use>) {
      result.buffers.at(buffer_index++) = declaration;
    } else if constexpr (std::same_as<Use, image_extent>) {
      result.image_extents.at(image_extent_index++) = declaration;
    } else {
      result.buffer_extents.at(buffer_extent_index++) = declaration;
    }
  };
  (append(declarations), ...);
  return result;
}

}// namespace vkexec

#endif// VKEXEC_RESOURCE_USE_HPP
