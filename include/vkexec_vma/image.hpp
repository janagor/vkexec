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
    [[nodiscard]] auto extent_3d() const noexcept -> VkExtent3D { return extent_3d_; }
    [[nodiscard]] auto mip_levels() const noexcept -> std::uint32_t { return mip_levels_; }
    [[nodiscard]] auto array_layers() const noexcept -> std::uint32_t { return array_layers_; }
    [[nodiscard]] auto type() const noexcept -> VkImageType { return type_; }
    [[nodiscard]] auto samples() const noexcept -> VkSampleCountFlagBits { return samples_; }
    [[nodiscard]] auto flags() const noexcept -> VkImageCreateFlags { return flags_; }
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
      ::vkexec::image_create_info const &info,
      image_usage usage,
      VkImageUsageFlags usage_flags) noexcept;

    auto destroy() noexcept -> void;

    VmaAllocator allocator_{ VK_NULL_HANDLE };
    VkImage image_{ VK_NULL_HANDLE };
    VmaAllocation allocation_{ VK_NULL_HANDLE };
    VkFormat format_{ VK_FORMAT_UNDEFINED };
    VkExtent2D extent_{};
    VkExtent3D extent_3d_{};
    std::uint32_t mip_levels_{ 1 };
    std::uint32_t array_layers_{ 1 };
    VkImageType type_{ VK_IMAGE_TYPE_2D };
    VkSampleCountFlagBits samples_{ VK_SAMPLE_COUNT_1_BIT };
    VkImageCreateFlags flags_{};
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
    //! Use this overload to select an explicit aspect, view type, or subresource range.
    [[nodiscard]] auto operator()(context &ctx, image const &img, ::vkexec::image_view_create_info info) const
    {
      if (info.format == VK_FORMAT_UNDEFINED) { info.format = img.format(); }
      return ::vkexec::factory::make_image_view(ctx, img.handle(), info);
    }

    //! Convenience for standard color and depth images. Use the explicit overload
    //! for stencil or combined depth/stencil formats and cube arrays.
    [[nodiscard]] auto operator()(context &ctx, image const &img) const
    {
      constexpr std::uint32_t k_cube_layers = 6;
      ::vkexec::image_view_create_info info{};
      info.range.aspectMask = img.usage() == image_usage::depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
      info.range.levelCount = img.mip_levels();
      info.range.layerCount = img.array_layers();
      if (img.type() == VK_IMAGE_TYPE_1D) {
        info.type = img.array_layers() == 1 ? VK_IMAGE_VIEW_TYPE_1D : VK_IMAGE_VIEW_TYPE_1D_ARRAY;
      } else if (img.type() == VK_IMAGE_TYPE_3D) {
        info.type = VK_IMAGE_VIEW_TYPE_3D;
      } else if ((img.flags() & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != 0 && img.array_layers() == k_cube_layers) {
        info.type = VK_IMAGE_VIEW_TYPE_CUBE;
      } else {
        info.type = img.array_layers() == 1 ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      }
      return (*this)(ctx, img, info);
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
