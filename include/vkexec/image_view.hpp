#ifndef VKEXEC_IMAGE_VIEW_HPP
#define VKEXEC_IMAGE_VIEW_HPP

#include <vkexec/context.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/image.hpp>

#include <vulkan/vulkan.h>

namespace vkexec {

/**
 * RAII `VkImageView` for a `vkexec::image`.
 *
 * The view does not own the image; `img` must outlive this view. Destroyed on
 * the context device when this object is destroyed or moved-from.
 *
 * @see image
 */
class image_view
{
public:
  /**
   * Creates a 2D image view matching `img`'s format and aspect.
   *
   * @param ctx Context that owns the device.
   * @param img Image to view (must remain alive while the view is used).
   */
  [[nodiscard]] static auto create(context &ctx, image const &img) -> detail::sync_sender_fn<image_view>;

  ~image_view();

  image_view(image_view const &) = delete;
  auto operator=(image_view const &) -> image_view & = delete;

  image_view(image_view &&other) noexcept;
  auto operator=(image_view &&other) noexcept -> image_view &;

  //! Vulkan image view handle (null after move).
  [[nodiscard]] auto handle() const noexcept -> VkImageView { return view_; }

private:
  image_view(context *ctx, VkImageView view) noexcept;
  auto destroy() noexcept -> void;

  context *ctx_{ nullptr };
  VkImageView view_{ VK_NULL_HANDLE };
};

}// namespace vkexec

#endif// VKEXEC_IMAGE_VIEW_HPP
