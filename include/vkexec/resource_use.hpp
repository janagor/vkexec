#ifndef VKEXEC_RESOURCE_USE_HPP
#define VKEXEC_RESOURCE_USE_HPP

//! \file
//! Resource declarations for custom pass recording.

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
  storage_compute,
  color_attachment,
  depth_attachment,
  transfer_source,
  transfer_destination,
};

enum class buffer_usage : std::uint8_t {
  vertex,
  index,
  uniform,
  storage_compute,
  transfer_source,
  transfer_destination,
};

struct image_use
{
  VkImage image{ VK_NULL_HANDLE };
  VkImageSubresourceRange range{};
  image_usage usage{};
  resource_access access{};
};

struct buffer_use
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  VkDeviceSize offset{};
  VkDeviceSize size{ VK_WHOLE_SIZE };
  buffer_usage usage{};
  resource_access access{};
};

[[nodiscard]] constexpr auto read(VkImage image, VkImageSubresourceRange range, image_usage usage) noexcept -> image_use
{ return { .image = image, .range = range, .usage = usage, .access = resource_access::read }; }

[[nodiscard]] constexpr auto write(VkImage image, VkImageSubresourceRange range, image_usage usage) noexcept
  -> image_use
{ return { .image = image, .range = range, .usage = usage, .access = resource_access::write }; }

[[nodiscard]] constexpr auto read_write(VkImage image, VkImageSubresourceRange range, image_usage usage) noexcept
  -> image_use
{ return { .image = image, .range = range, .usage = usage, .access = resource_access::read_write }; }

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

template<resource_use... Uses> struct resource_uses
{
  std::array<image_use, (std::size_t{} + ... + std::size_t{ std::same_as<Uses, image_use> })> images{};
  std::array<buffer_use, (std::size_t{} + ... + std::size_t{ std::same_as<Uses, buffer_use> })> buffers{};
};

template<resource_use... Uses> [[nodiscard]] constexpr auto uses(Uses... declarations) -> resource_uses<Uses...>
{
  resource_uses<Uses...> result{};
  std::size_t image_index{};
  std::size_t buffer_index{};
  auto append = [&]<resource_use Use>(Use declaration) constexpr -> void {
    if constexpr (std::same_as<Use, image_use>) {
      result.images.at(image_index++) = declaration;
    } else {
      result.buffers.at(buffer_index++) = declaration;
    }
  };
  (append(declarations), ...);
  return result;
}

}// namespace vkexec

#endif// VKEXEC_RESOURCE_USE_HPP
