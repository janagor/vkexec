#ifndef VKEXEC_VMA_IMAGE_HPP
#define VKEXEC_VMA_IMAGE_HPP

//! \file
//! Untyped VMA images for offscreen (non-swapchain) targets.

#include <vkexec/error.hpp>
#include <vkexec/image_view.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/sender.hpp>
#include <vkexec_vma/allocator.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstdint>

namespace vkexec {

namespace vma {

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
   * Creation parameters for `factory::make_image`.
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

}// namespace vma

namespace vma {
  class image;
}// namespace vma

namespace detail {

  struct make_image_factory
  {
    vma::allocator *allocator;
    vma::image_create_info info;

    [[nodiscard]] auto operator()() const -> result<vma::image>;
  };

  struct make_generic_image_factory
  {
    vma::allocator *allocator;
    ::vkexec::image_create_info info;

    [[nodiscard]] auto operator()() const -> result<vma::image>;
  };

}// namespace detail

/**
 * Untyped VMA image for offscreen targets (not swapchain images).
 *
 * Move-only; destroys via VMA when owned. Create views with `image_view`.
 *
 * @see image_view, factory::make_image, image_create_info
 */
namespace vma {

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
    //! Depth or color aspect category used by the image-view convenience factory.
    [[nodiscard]] auto usage() const noexcept -> image_usage { return usage_; }
    //! Exact Vulkan usage flags used at creation.
    [[nodiscard]] auto usage_flags() const noexcept -> VkImageUsageFlags { return usage_flags_; }

  private:
    friend struct ::vkexec::detail::make_image_factory;
    friend struct ::vkexec::detail::make_generic_image_factory;

    image(VmaAllocator allocator_handle,
      VkImage image_handle,
      VmaAllocation allocation,
      VkFormat format,
      VkExtent2D extent,
      image_usage usage,
      VkImageUsageFlags usage_flags) noexcept;

    auto destroy() noexcept -> void;

    VmaAllocator allocator_{ VK_NULL_HANDLE };
    VkImage image_{ VK_NULL_HANDLE };
    VmaAllocation allocation_{ VK_NULL_HANDLE };
    VkFormat format_{ VK_FORMAT_UNDEFINED };
    VkExtent2D extent_{};
    image_usage usage_{ image_usage::color_storage };
    VkImageUsageFlags usage_flags_{};
  };

}// namespace vma

}// namespace vkexec

namespace vkexec::vma {

namespace factory {
  struct make_image_t
  {
    [[nodiscard]] auto operator()(allocator &allocator, image_create_info info) const
    { return make_sender(::vkexec::detail::make_image_factory{ .allocator = &allocator, .info = info }); }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_image_t make_image{};

  // Generic allocation requests retain the exact owning VMA image type.

  struct make_image_view_t
  {
    [[nodiscard]] auto operator()(context &ctx, image const &img) const
    {
      return ::vkexec::factory::make_image_view(ctx,
        img.handle(),
        img.format(),
        img.usage() == image_usage::depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
    }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_image_view_t make_image_view{};
}// namespace factory

[[nodiscard]] inline auto tag_invoke(allocate_image_t /*tag*/, allocator &alloc, ::vkexec::image_create_info info)
{ return make_sender(::vkexec::detail::make_generic_image_factory{ .allocator = &alloc, .info = info }); }

static_assert(image_allocator<allocator>);

}// namespace vkexec::vma

#endif// VKEXEC_VMA_IMAGE_HPP
