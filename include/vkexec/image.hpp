#ifndef VKEXEC_IMAGE_HPP
#define VKEXEC_IMAGE_HPP

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstdint>

namespace vkexec {

enum class image_usage : std::uint8_t {
  /// Device-local color target usable as storage image and color attachment.
  color_storage,
  /// Device-local depth attachment.
  depth,
};

struct image_create_info
{
  std::uint32_t width{ 1 };
  std::uint32_t height{ 1 };
  image_usage usage{ image_usage::color_storage };
  VkFormat format{ VK_FORMAT_UNDEFINED };
};

/// Untyped VMA image for offscreen targets (non-swapchain).
class image
{
public:
  [[nodiscard]] static auto create(context &ctx, image_create_info info) -> result<image>;

  ~image();

  image(image const &) = delete;
  auto operator=(image const &) -> image & = delete;

  image(image &&other) noexcept;
  auto operator=(image &&other) noexcept -> image &;

  [[nodiscard]] auto handle() const noexcept -> VkImage { return image_; }
  [[nodiscard]] auto format() const noexcept -> VkFormat { return format_; }
  [[nodiscard]] auto extent() const noexcept -> VkExtent2D { return extent_; }
  [[nodiscard]] auto usage() const noexcept -> image_usage { return usage_; }

private:
  image(context *ctx,
    VkImage image,
    VmaAllocation allocation,
    VkFormat format,
    VkExtent2D extent,
    image_usage usage) noexcept;

  auto destroy() noexcept -> void;

  context *ctx_{ nullptr };
  VkImage image_{ VK_NULL_HANDLE };
  VmaAllocation allocation_{ VK_NULL_HANDLE };
  VkFormat format_{ VK_FORMAT_UNDEFINED };
  VkExtent2D extent_{};
  image_usage usage_{ image_usage::color_storage };
};

}// namespace vkexec

#endif// VKEXEC_IMAGE_HPP
