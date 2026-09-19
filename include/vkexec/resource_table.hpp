#ifndef VKEXEC_RESOURCE_TABLE_HPP
#define VKEXEC_RESOURCE_TABLE_HPP

//! \file
//! Heap-agnostic logical resource bindings.

#include <vulkan/vulkan_core.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

//! Descriptor resource category shared by schemas and runtime tables.
enum class resource_kind : std::uint8_t { storage_buffer, storage_image, sampled_image, sampler };

//! One descriptor resource without descriptor-mechanism metadata.
struct resource_ref
{
  resource_kind kind{ resource_kind::storage_buffer };
  VkBuffer buffer{ VK_NULL_HANDLE };
  VkDeviceSize byte_size{ 0 };
  VkImageView image_view{ VK_NULL_HANDLE };
  VkImageLayout image_layout{ VK_IMAGE_LAYOUT_GENERAL };
  VkSampler sampler{ VK_NULL_HANDLE };
};

//! Creates a storage-buffer resource reference.
[[nodiscard]] inline auto buffer_resource(VkBuffer buffer, VkDeviceSize byte_size) noexcept -> resource_ref
{
  return resource_ref{ .kind = resource_kind::storage_buffer,
    .buffer = buffer,
    .byte_size = byte_size,
    .image_view = VK_NULL_HANDLE,
    .image_layout = VK_IMAGE_LAYOUT_GENERAL,
    .sampler = VK_NULL_HANDLE };
}

//! Creates a storage-image resource reference.
[[nodiscard]] inline auto storage_image_resource(
  VkImageView image_view, VkImageLayout image_layout = VK_IMAGE_LAYOUT_GENERAL) noexcept -> resource_ref
{
  return resource_ref{ .kind = resource_kind::storage_image,
    .buffer = VK_NULL_HANDLE,
    .byte_size = 0,
    .image_view = image_view,
    .image_layout = image_layout,
    .sampler = VK_NULL_HANDLE };
}

//! Creates a sampled-image resource reference.
[[nodiscard]] inline auto sampled_image_resource(
  VkImageView image_view, VkImageLayout image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) noexcept
  -> resource_ref
{
  return resource_ref{ .kind = resource_kind::sampled_image,
    .buffer = VK_NULL_HANDLE,
    .byte_size = 0,
    .image_view = image_view,
    .image_layout = image_layout,
    .sampler = VK_NULL_HANDLE };
}

//! Creates a sampler resource reference.
[[nodiscard]] inline auto sampler_resource(VkSampler sampler) noexcept -> resource_ref
{
  return resource_ref{ .kind = resource_kind::sampler,
    .buffer = VK_NULL_HANDLE,
    .byte_size = 0,
    .image_view = VK_NULL_HANDLE,
    .image_layout = VK_IMAGE_LAYOUT_GENERAL,
    .sampler = sampler };
}

//! Associates a logical shader slot with a resource.
struct resource_binding
{
  std::uint32_t slot{ 0 };
  resource_ref resource{};
};

//! Ordered bag of logical resource bindings lowered by a descriptor table backend.
class resource_table
{
public:
  resource_table() = default;
  explicit resource_table(std::vector<resource_binding> entries) : entries_(std::move(entries)) {}

  [[nodiscard]] auto entries() const noexcept -> std::span<resource_binding const> { return entries_; }
  [[nodiscard]] auto size() const noexcept -> std::size_t { return entries_.size(); }
  [[nodiscard]] auto empty() const noexcept -> bool { return entries_.empty(); }

private:
  std::vector<resource_binding> entries_;
};

//! Writes all table resource kinds into a classic descriptor set.
auto write_resource_descriptors(VkDevice device, VkDescriptorSet set, std::span<resource_binding const> resources)
  -> void;

//! Creates a resource table while preserving argument order.
template<class... Bindings>
  requires(std::same_as<std::remove_cvref_t<Bindings>, resource_binding> && ...)
[[nodiscard]] auto bindings(Bindings &&...entries) -> resource_table
{
  std::vector<resource_binding> resources;
  resources.reserve(sizeof...(Bindings));
  (resources.push_back(std::forward<Bindings>(entries)), ...);
  return resource_table{ std::move(resources) };
}

}// namespace vkexec

#endif// VKEXEC_RESOURCE_TABLE_HPP
