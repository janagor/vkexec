#ifndef VKEXEC_IMAGE_VIEW_HPP
#define VKEXEC_IMAGE_VIEW_HPP

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/sender.hpp>

#include <vulkan/vulkan.h>

namespace vkexec {

struct image_view_create_info
{
  VkImageViewType type{ VK_IMAGE_VIEW_TYPE_2D };
  VkFormat format{ VK_FORMAT_UNDEFINED };
  VkImageSubresourceRange range{
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = 1,
    .baseArrayLayer = 0,
    .layerCount = 1,
  };
  VkComponentMapping components{};
};

namespace owned {
  class image_view;
}// namespace owned

namespace detail {

  struct make_image_view_factory
  {
    context *ctx;
    VkImage image;
    image_view_create_info info;

    [[nodiscard]] auto operator()() const -> result<owned::image_view>;
  };

}// namespace detail

namespace factory {

  struct make_image_view_t
  {
    [[nodiscard]] auto operator()(context &ctx, VkImage image, image_view_create_info info) const
    { return make_sender(detail::make_image_view_factory{ .ctx = &ctx, .image = image, .info = info }); }

    /**
     * Creates a 2D image view for raw Vulkan image facts.
     *
     * @param ctx Context that owns the device.
     * @param image Image to view (must remain alive while the view is used).
     */
    [[nodiscard]] auto operator()(context &ctx, VkImage image, VkFormat format, VkImageAspectFlags aspect) const
    {
      image_view_create_info info{};
      info.format = format;
      info.range.aspectMask = aspect;
      return (*this)(ctx, image, info);
    }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_image_view_t make_image_view{};

}// namespace factory

/**
 * RAII `VkImageView` for a Vulkan image.
 *
 * The view does not own the image; it must outlive this view. Destroyed on
 * the context device when this object is destroyed or moved-from.
 *
 * @see image, factory::make_image_view
 */
namespace owned {

  class image_view
  {
  public:
    ~image_view();

    image_view(image_view const &) = delete;
    auto operator=(image_view const &) -> image_view & = delete;

    image_view(image_view &&other) noexcept;
    auto operator=(image_view &&other) noexcept -> image_view &;

    //! Vulkan image view handle (null after move).
    [[nodiscard]] auto handle() const noexcept -> VkImageView { return view_; }

  private:
    friend struct detail::make_image_view_factory;

    image_view(context *ctx, VkImageView view) noexcept;
    auto destroy() noexcept -> void;

    context *ctx_{ nullptr };
    VkImageView view_{ VK_NULL_HANDLE };
  };

}// namespace owned

}// namespace vkexec

#endif// VKEXEC_IMAGE_VIEW_HPP
