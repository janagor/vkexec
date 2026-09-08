#include <vkexec_graphics/swapchain.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>

#include <VkBootstrap.h>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace vkexec {

auto swapchain::create(context &ctx, swapchain_create_info info) -> result<swapchain>
{
  if (ctx.device() == VK_NULL_HANDLE) { return fail(errc::invalid_argument, "swapchain requires a VkDevice"); }
  if (info.surface == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "swapchain requires a VkSurfaceKHR");
  }
  if (ctx.present_queue() == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "swapchain requires a present queue");
  }

  swapchain created;
  created.ctx_ = &ctx;
  created.surface_ = info.surface;
  created.preferred_format_ = info.preferred_format;
  created.preferred_color_space_ = info.preferred_color_space;
  created.present_mode_ = info.present_mode;
  if (auto created_swapchain = created.create_or_recreate(info.width, info.height); !created_swapchain) {
    return fail(created_swapchain);
  }
  return created;
}

swapchain::~swapchain() { destroy(); }

swapchain::swapchain(swapchain &&other) noexcept
  : ctx_(other.ctx_), surface_(other.surface_), preferred_format_(other.preferred_format_),
    preferred_color_space_(other.preferred_color_space_), present_mode_(other.present_mode_),
    swapchain_(other.swapchain_), format_(other.format_), extent_(other.extent_), images_(std::move(other.images_)),
    views_(std::move(other.views_))
{
  other.ctx_ = nullptr;
  other.surface_ = VK_NULL_HANDLE;
  other.swapchain_ = {};
}

auto swapchain::operator=(swapchain &&other) noexcept -> swapchain &
{
  if (this == &other) { return *this; }
  destroy();
  ctx_ = other.ctx_;
  surface_ = other.surface_;
  preferred_format_ = other.preferred_format_;
  preferred_color_space_ = other.preferred_color_space_;
  present_mode_ = other.present_mode_;
  swapchain_ = other.swapchain_;
  format_ = other.format_;
  extent_ = other.extent_;
  images_ = std::move(other.images_);
  views_ = std::move(other.views_);
  other.ctx_ = nullptr;
  other.surface_ = VK_NULL_HANDLE;
  other.swapchain_ = {};
  return *this;
}

auto swapchain::recreate(std::uint32_t width, std::uint32_t height) -> status
{ return create_or_recreate(width, height); }

auto swapchain::acquire_next_image(VkSemaphore image_available, std::uint64_t timeout)
  -> result<std::optional<std::uint32_t>>
{
  if (ctx_ == nullptr || swapchain_.swapchain == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "swapchain::acquire_next_image on empty swapchain");
  }

  std::uint32_t image_index = 0;
  VkResult const result =
    vkAcquireNextImageKHR(ctx_->device(), swapchain_.swapchain, timeout, image_available, VK_NULL_HANDLE, &image_index);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) { return std::optional<std::uint32_t>{}; }
  if (result != VK_SUCCESS) { return fail(result, "vkAcquireNextImageKHR failed"); }
  return image_index;
}

auto swapchain::present(std::uint32_t image_index, std::span<VkSemaphore const> wait_semaphores) -> result<bool>
{
  if (ctx_ == nullptr || swapchain_.swapchain == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "swapchain::present on empty swapchain");
  }

  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = static_cast<std::uint32_t>(wait_semaphores.size());
  present_info.pWaitSemaphores = wait_semaphores.empty() ? nullptr : wait_semaphores.data();
  present_info.swapchainCount = 1;
  present_info.pSwapchains = &swapchain_.swapchain;
  present_info.pImageIndices = &image_index;

  VkResult const result = vkQueuePresentKHR(ctx_->present_queue(), &present_info);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) { return false; }
  if (result != VK_SUCCESS) { return fail(result, "vkQueuePresentKHR failed"); }
  return true;
}

auto swapchain::create_or_recreate(std::uint32_t width, std::uint32_t height) -> status
{
  if (width == 0 || height == 0) { return fail(errc::invalid_argument, "swapchain extent must be > 0"); }

  destroy_views();
  images_.clear();

  vkb::Swapchain const old_swapchain = swapchain_;
  auto builder = vkb::SwapchainBuilder{ ctx_->vkb_device(), surface_ }
                   .set_desired_format({ .format = preferred_format_, .colorSpace = preferred_color_space_ })
                   .set_desired_present_mode(present_mode_)
                   .set_desired_extent(width, height)
                   .set_old_swapchain(old_swapchain);

  auto const built = builder.build();
  if (!built) { return fail(make_error_from_vkb(built, "vk-bootstrap SwapchainBuilder")); }
  swapchain_ = built.value();
  if (old_swapchain.swapchain != VK_NULL_HANDLE) { vkb::destroy_swapchain(old_swapchain); }

  format_ = swapchain_.image_format;
  extent_ = swapchain_.extent;

  auto const got_images = swapchain_.get_images();
  if (!got_images) { return fail(make_error_from_vkb(got_images, "vk-bootstrap Swapchain::get_images")); }
  images_ = got_images.value();

  auto const got_views = swapchain_.get_image_views();
  if (!got_views) { return fail(make_error_from_vkb(got_views, "vk-bootstrap Swapchain::get_image_views")); }
  views_ = got_views.value();
  return {};
}

auto swapchain::destroy_views() noexcept -> void
{
  if (ctx_ == nullptr || swapchain_.swapchain == VK_NULL_HANDLE || views_.empty()) {
    views_.clear();
    return;
  }
  swapchain_.destroy_image_views(views_);
  views_.clear();
}

auto swapchain::destroy() noexcept -> void
{
  destroy_views();
  images_.clear();
  if (swapchain_.swapchain != VK_NULL_HANDLE) {
    vkb::destroy_swapchain(swapchain_);
    swapchain_ = {};
  }
  ctx_ = nullptr;
  surface_ = VK_NULL_HANDLE;
}

}// namespace vkexec
