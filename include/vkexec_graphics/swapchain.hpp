#ifndef VKEXEC_GRAPHICS_SWAPCHAIN_HPP
#define VKEXEC_GRAPHICS_SWAPCHAIN_HPP

#include <vkexec/context.hpp>

#include <VkBootstrap.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace vkexec {

struct swapchain_create_info
{
  VkSurfaceKHR surface{ VK_NULL_HANDLE };
  std::uint32_t width{ 0 };
  std::uint32_t height{ 0 };
  VkFormat preferred_format{ VK_FORMAT_B8G8R8A8_SRGB };
  VkColorSpaceKHR preferred_color_space{ VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
  VkPresentModeKHR present_mode{ VK_PRESENT_MODE_FIFO_KHR };
};

/// Presentable swapchain that borrows a surface (does not destroy it).
/// Suitable for embedders that own the native window / surface separately.
class swapchain
{
public:
  [[nodiscard]] static auto create(context &ctx, swapchain_create_info info) -> swapchain;

  ~swapchain();

  swapchain(swapchain const &) = delete;
  auto operator=(swapchain const &) -> swapchain & = delete;

  swapchain(swapchain &&other) noexcept;
  auto operator=(swapchain &&other) noexcept -> swapchain &;

  /// Recreate for a new extent. Destroys old image views and swapchain images.
  auto recreate(std::uint32_t width, std::uint32_t height) -> void;

  /// Acquire the next image. Returns nullopt when the swapchain must be recreated.
  [[nodiscard]] auto acquire_next_image(VkSemaphore image_available, std::uint64_t timeout = UINT64_MAX)
    -> std::optional<std::uint32_t>;

  /// Present `image_index`. Returns false when the swapchain must be recreated.
  [[nodiscard]] auto present(std::uint32_t image_index, std::span<VkSemaphore const> wait_semaphores) -> bool;

  [[nodiscard]] auto handle() const noexcept -> VkSwapchainKHR { return swapchain_.swapchain; }
  [[nodiscard]] auto format() const noexcept -> VkFormat { return format_; }
  [[nodiscard]] auto extent() const noexcept -> VkExtent2D { return extent_; }
  [[nodiscard]] auto images() const noexcept -> std::span<VkImage const> { return images_; }
  [[nodiscard]] auto image_views() const noexcept -> std::span<VkImageView const> { return views_; }
  [[nodiscard]] auto surface() const noexcept -> VkSurfaceKHR { return surface_; }

private:
  swapchain(context *ctx, VkSurfaceKHR surface, swapchain_create_info info);

  auto create_or_recreate(std::uint32_t width, std::uint32_t height) -> void;
  auto destroy_views() noexcept -> void;
  auto destroy() noexcept -> void;

  context *ctx_{ nullptr };
  VkSurfaceKHR surface_{ VK_NULL_HANDLE };
  VkFormat preferred_format_{ VK_FORMAT_B8G8R8A8_SRGB };
  VkColorSpaceKHR preferred_color_space_{ VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
  VkPresentModeKHR present_mode_{ VK_PRESENT_MODE_FIFO_KHR };

  vkb::Swapchain swapchain_{};
  VkFormat format_{ VK_FORMAT_B8G8R8A8_SRGB };
  VkExtent2D extent_{};
  std::vector<VkImage> images_;
  std::vector<VkImageView> views_;
};

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_SWAPCHAIN_HPP
