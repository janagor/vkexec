#ifndef VKEXEC_RESOURCE_ALLOCATOR_HPP
#define VKEXEC_RESOURCE_ALLOCATOR_HPP

//! \file
//! Static Vulkan resource allocation protocol. Execution itself uses raw handles.

#include <vkexec/sender.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

class context;

enum class memory_domain : std::uint8_t { host_visible, device_local, staging };

struct buffer_create_info
{
  VkDeviceSize size{};
  VkBufferUsageFlags usage{};
  memory_domain memory{ memory_domain::host_visible };
  bool shader_device_address{};
  VkDeviceSize alignment{};
};

struct image_create_info
{
  // The allocator checks basic values. Other image-type, extent, sample-count,
  // and creation-flag combinations are passed through to Vulkan/VMA.
  VkExtent3D extent{};
  VkFormat format{ VK_FORMAT_UNDEFINED };
  VkImageUsageFlags usage{};
  VkImageType type{ VK_IMAGE_TYPE_2D };
  std::uint32_t mip_levels{ 1 };
  std::uint32_t array_layers{ 1 };
  VkSampleCountFlagBits samples{ VK_SAMPLE_COUNT_1_BIT };
  VkImageTiling tiling{ VK_IMAGE_TILING_OPTIMAL };
  VkImageCreateFlags flags{};
  //! Concurrent sharing requires at least two family indices; the sender owns this copy.
  VkSharingMode sharing_mode{ VK_SHARING_MODE_EXCLUSIVE };
  std::vector<std::uint32_t> queue_families;
};

// Customizations are found through ADL and keep the backend's allocation token typed.
void tag_invoke();

struct allocate_buffer_t
{
  template<class A>
    requires requires(A &allocator, buffer_create_info info) {
      tag_invoke(std::declval<allocate_buffer_t>(), allocator, info);
    }
  [[nodiscard]] auto operator()(A &allocator, buffer_create_info info) const
    -> decltype(tag_invoke(*this, allocator, info))
  { return tag_invoke(*this, allocator, info); }
};

struct allocate_image_t
{
  template<class A>
    requires requires(A &allocator, image_create_info info) {
      tag_invoke(std::declval<allocate_image_t>(), allocator, info);
    }
  [[nodiscard]] auto operator()(A &allocator, image_create_info info) const
    -> decltype(tag_invoke(*this, allocator, info))
  { return tag_invoke(*this, allocator, std::move(info)); }
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr allocate_buffer_t allocate_buffer{};
// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr allocate_image_t allocate_image{};

template<class B>
concept owning_buffer_resource = std::movable<B> && requires(B const &buffer) {
  { buffer.handle() } -> std::same_as<VkBuffer>;
  { buffer.size() } -> std::convertible_to<VkDeviceSize>;
};

template<class B>
concept mapped_buffer_resource = owning_buffer_resource<B> && requires(B &buffer, B const &cbuffer) {
  { buffer.mapped() } -> std::convertible_to<std::span<std::byte>>;
  { cbuffer.mapped() } -> std::convertible_to<std::span<std::byte const>>;
};

template<class B>
concept flushable_buffer_resource = mapped_buffer_resource<B> && requires(B &buffer) {
  { buffer.flush() } -> std::same_as<status>;
};

template<class B>
concept readback_buffer_resource = flushable_buffer_resource<B> && requires(B &buffer) {
  { buffer.invalidate() } -> std::same_as<status>;
};

template<class B>
concept device_addressable_buffer_resource = owning_buffer_resource<B> && requires(B const &buffer) {
  { buffer.device_address() } -> std::same_as<result<VkDeviceAddress>>;
};

template<class I>
concept image_resource = std::movable<I> && requires(I const &image) {
  { image.handle() } -> std::same_as<VkImage>;
  { image.format() } -> std::same_as<VkFormat>;
};

template<class Sender, class Resource>
concept resource_allocation_sender = vkexec_sender_of<Sender, Resource>;

template<class A>
concept buffer_allocator = requires(A &allocator, buffer_create_info info) {
  typename A::buffer_type;
  requires owning_buffer_resource<typename A::buffer_type>;
  requires resource_allocation_sender<decltype(allocate_buffer(allocator, info)), typename A::buffer_type>;
};

template<class A>
concept image_allocator = requires(A &allocator, image_create_info info) {
  typename A::image_type;
  requires image_resource<typename A::image_type>;
  requires resource_allocation_sender<decltype(allocate_image(allocator, info)), typename A::image_type>;
};

template<class A>
concept resource_allocator = buffer_allocator<A> && image_allocator<A>;

}// namespace vkexec

#endif// VKEXEC_RESOURCE_ALLOCATOR_HPP
