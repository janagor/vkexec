#ifndef VKEXEC_IMAGE_VIEW_HPP
#define VKEXEC_IMAGE_VIEW_HPP

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/image.hpp>
#include <vkexec/sender.hpp>

#include <vulkan/vulkan.h>

namespace vkexec {

namespace owned {
class image_view;
}// namespace owned

namespace factory {

  struct make_image_view_t
  {

    /**
     * Creates a 2D image view matching `img`'s format and aspect.
     *
     * @param ctx Context that owns the device.
     * @param img Image to view (must remain alive while the view is used).
     */
    [[nodiscard]] auto operator()(context &ctx, owned::image const &img) const -> sender<owned::image_view>;
  };

  //NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_image_view_t make_image_view{};

}// namespace factory

/**
 * RAII `VkImageView` for an `owned::image`.
 *
 * The view does not own the image; `img` must outlive this view. Destroyed on
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
  friend struct factory::make_image_view_t;

  image_view(context *ctx, VkImageView view) noexcept;
  auto destroy() noexcept -> void;

  context *ctx_{ nullptr };
  VkImageView view_{ VK_NULL_HANDLE };
};

}// namespace owned

}// namespace vkexec

#endif// VKEXEC_IMAGE_VIEW_HPP
