#ifndef VKEXEC_IMAGE_HPP
#define VKEXEC_IMAGE_HPP

//! \file
//! Untyped VMA images for offscreen (non-swapchain) targets.

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/sender.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstdint>

namespace vkexec {

/**
 * Usage preset for offscreen `image` allocations.
 *
 * @see image_create_info, image
 */
enum class image_usage : std::uint8_t {
  //! Device-local color target usable as storage image and color attachment.
  color_storage,
  //! Device-local depth attachment.
  depth,
};

/**
 * Creation parameters for `factory::image`.
 *
 * When `format` is `VK_FORMAT_UNDEFINED`, a default format for `usage` is chosen.
 */
struct image_create_info
{
  std::uint32_t width{ 1 };
  std::uint32_t height{ 1 };
  image_usage usage{ image_usage::color_storage };
  VkFormat format{ VK_FORMAT_UNDEFINED };
};

class image;

namespace factory {

  /**
   * Creates a device-local image described by `info`.
   *
   * @param ctx Context whose VMA allocator owns the allocation.
   * @param info Extent, usage, and optional format override.
   */
  [[nodiscard]] auto image(::vkexec::context &ctx, image_create_info info) -> sender<::vkexec::image>;

}// namespace factory

/**
 * Untyped VMA image for offscreen targets (not swapchain images).
 *
 * Move-only; destroys via VMA when owned. Create views with `image_view`.
 *
 * @see image_view, factory::image, image_create_info
 */
class image
{
public:
  ~image();

  image(image const &) = delete;
  auto operator=(image const &) -> image & = delete;

  image(image &&other) noexcept;
  auto operator=(image &&other) noexcept -> image &;

  //! Vulkan image handle (null after move).
  [[nodiscard]] auto handle() const noexcept -> VkImage { return image_; }
  //! Image format chosen at creation.
  [[nodiscard]] auto format() const noexcept -> VkFormat { return format_; }
  //! Image extent in pixels.
  [[nodiscard]] auto extent() const noexcept -> VkExtent2D { return extent_; }
  //! Usage preset used at creation.
  [[nodiscard]] auto usage() const noexcept -> image_usage { return usage_; }

private:
  friend auto factory::image(::vkexec::context &ctx, image_create_info info) -> sender<::vkexec::image>;

  image(context *ctx,
    VkImage image_handle,
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
