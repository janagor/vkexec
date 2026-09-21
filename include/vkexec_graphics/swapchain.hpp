#ifndef VKEXEC_GRAPHICS_SWAPCHAIN_HPP
#define VKEXEC_GRAPHICS_SWAPCHAIN_HPP

//! \file
//! Presentable Vulkan swapchain that borrows an external surface.

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>

#include <VkBootstrap.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace vkexec {

/**
 * Optional extension chain for `VkPresentInfoKHR`.
 *
 * The pointed chain only needs to remain valid for the duration of `present`.
 */
struct present_options
{
  void const *p_next{ nullptr };
};

/**
 * Creation parameters for `factory::swapchain`.
 *
 * `surface` is borrowed and never destroyed by the swapchain.
 */
struct swapchain_create_info
{
  VkSurfaceKHR surface{ VK_NULL_HANDLE };
  std::uint32_t width{ 0 };
  std::uint32_t height{ 0 };
  VkFormat preferred_format{ VK_FORMAT_B8G8R8A8_SRGB };
  VkColorSpaceKHR preferred_color_space{ VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
  VkPresentModeKHR present_mode{ VK_PRESENT_MODE_FIFO_KHR };
  VkSwapchainCreateFlagsKHR flags{ 0 };
};

class swapchain;

namespace factory {

  /**
   * Creates a swapchain for `info.surface` at the given extent.
   *
   * @param ctx Context with presentation queues enabled.
   * @param info Surface, extent, and present preferences.
   */
  [[nodiscard]] auto swapchain(::vkexec::context &ctx, swapchain_create_info info) -> sender<::vkexec::swapchain>;

}// namespace factory

/**
 * Presentable swapchain that borrows a surface (does not destroy it).
 *
 * Suitable for embedders that own the native presenter / surface separately.
 * Prefer `presenter` when vkexec should own the complete presentation stack.
 *
 * @see presenter, acquire_present_frame, swapchain_create_info, factory::swapchain
 */
class swapchain
{
public:
  ~swapchain();

  swapchain(swapchain const &) = delete;
  auto operator=(swapchain const &) -> swapchain & = delete;

  swapchain(swapchain &&other) noexcept;
  auto operator=(swapchain &&other) noexcept -> swapchain &;

  /**
   * Recreates the swapchain for a new extent.
   *
   * Destroys old image views and swapchain images.
   *
   * @param width New width in pixels.
   * @param height New height in pixels.
   */
  auto recreate(std::uint32_t width, std::uint32_t height) -> status;

  /**
   * Acquires the next image.
   *
   * @param image_available Binary semaphore signalled when the image is ready.
   * @param timeout Acquire timeout in nanoseconds.
   * @return Image index, disengaged optional when recreate is required, or an error.
   */
  [[nodiscard]] auto acquire_next_image(VkSemaphore image_available, std::uint64_t timeout = UINT64_MAX)
    -> result<std::optional<std::uint32_t>>;

  /**
   * Presents `image_index`.
   *
   * @param image_index Swapchain image index from acquire.
   * @param wait_semaphores Semaphores to wait on before present.
   * @param options Optional `VkPresentInfoKHR::pNext` chain.
   * @return `false` when the swapchain must be recreated; `true` on success.
   */
  [[nodiscard]] auto present(std::uint32_t image_index,
    std::span<VkSemaphore const> wait_semaphores,
    present_options options = {}) -> result<bool>;

  //! Vulkan swapchain handle.
  [[nodiscard]] auto handle() const noexcept -> VkSwapchainKHR { return swapchain_.swapchain; }
  //! Chosen surface format.
  [[nodiscard]] auto format() const noexcept -> VkFormat { return format_; }
  //! Current swapchain extent.
  [[nodiscard]] auto extent() const noexcept -> VkExtent2D { return extent_; }
  //! Swapchain images (owned by the swapchain).
  [[nodiscard]] auto images() const noexcept -> std::span<VkImage const> { return images_; }
  //! Image views for `images()` (owned by this object).
  [[nodiscard]] auto image_views() const noexcept -> std::span<VkImageView const> { return views_; }
  //! Borrowed surface handle.
  [[nodiscard]] auto surface() const noexcept -> VkSurfaceKHR { return surface_; }

private:
  friend auto factory::swapchain(::vkexec::context &ctx, swapchain_create_info info) -> sender<::vkexec::swapchain>;

  swapchain() = default;

  auto create_or_recreate(std::uint32_t width, std::uint32_t height) -> status;
  auto destroy_views() noexcept -> void;
  auto destroy() noexcept -> void;

  context *ctx_{ nullptr };
  VkSurfaceKHR surface_{ VK_NULL_HANDLE };
  VkFormat preferred_format_{ VK_FORMAT_B8G8R8A8_SRGB };
  VkColorSpaceKHR preferred_color_space_{ VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
  VkPresentModeKHR present_mode_{ VK_PRESENT_MODE_FIFO_KHR };
  VkSwapchainCreateFlagsKHR create_flags_{ 0 };

  vkb::Swapchain swapchain_{};
  VkFormat format_{ VK_FORMAT_B8G8R8A8_SRGB };
  VkExtent2D extent_{};
  std::vector<VkImage> images_;
  std::vector<VkImageView> views_;
};

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_SWAPCHAIN_HPP
