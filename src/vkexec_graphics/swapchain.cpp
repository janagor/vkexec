#include <vkexec_graphics/swapchain.hpp>

#include <vkexec/context.hpp>
#include <vkexec/detail/vk_bootstrap_error.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/result.hpp>

#include <VkBootstrap.h>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace vkexec {

struct owned::swapchain::impl
{
  vkb::Swapchain swapchain{};
};

owned::swapchain::swapchain() = default;

auto owned::swapchain::handle() const noexcept -> VkSwapchainKHR
{ return impl_ ? impl_->swapchain.swapchain : VK_NULL_HANDLE; }

auto detail::make_swapchain_factory::operator()() const -> result<::vkexec::owned::swapchain>
{
  if (ctx->device() == VK_NULL_HANDLE) { return fail(errc::invalid_argument, "swapchain requires a VkDevice"); }
  if (info.surface == VK_NULL_HANDLE) { return fail(errc::invalid_argument, "swapchain requires a VkSurfaceKHR"); }
  if (ctx->present_queue() == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "swapchain requires a present queue");
  }

  ::vkexec::owned::swapchain created;
  created.ctx_ = ctx;
  created.surface_ = info.surface;
  created.preferred_format_ = info.preferred_format;
  created.preferred_color_space_ = info.preferred_color_space;
  created.present_mode_ = info.present_mode;
  created.create_flags_ = info.flags;
  if (auto created_swapchain = created.create_or_recreate(info.width, info.height); !created_swapchain) {
    return fail(created_swapchain);
  }
  return created;
}

owned::swapchain::~swapchain() { destroy(); }

owned::swapchain::swapchain(swapchain &&other) noexcept
  : ctx_(other.ctx_), surface_(other.surface_), preferred_format_(other.preferred_format_),
    preferred_color_space_(other.preferred_color_space_), present_mode_(other.present_mode_),
    create_flags_(other.create_flags_), impl_(std::move(other.impl_)), format_(other.format_), extent_(other.extent_),
    images_(std::move(other.images_)), views_(std::move(other.views_))
{
  other.ctx_ = nullptr;
  other.surface_ = VK_NULL_HANDLE;
}

auto owned::swapchain::operator=(swapchain &&other) noexcept -> swapchain &
{
  if (this == &other) { return *this; }
  destroy();
  ctx_ = other.ctx_;
  surface_ = other.surface_;
  preferred_format_ = other.preferred_format_;
  preferred_color_space_ = other.preferred_color_space_;
  present_mode_ = other.present_mode_;
  create_flags_ = other.create_flags_;
  impl_ = std::move(other.impl_);
  format_ = other.format_;
  extent_ = other.extent_;
  images_ = std::move(other.images_);
  views_ = std::move(other.views_);
  other.ctx_ = nullptr;
  other.surface_ = VK_NULL_HANDLE;
  return *this;
}

auto owned::swapchain::recreate(std::uint32_t width, std::uint32_t height) -> status
{ return create_or_recreate(width, height); }

auto owned::swapchain::acquire_next_image(VkSemaphore image_available, std::uint64_t timeout)
  -> result<std::optional<std::uint32_t>>
{
  if (ctx_ == nullptr || handle() == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "owned::swapchain::acquire_next_image on empty swapchain");
  }

  std::uint32_t image_index = 0;
  VkResult const result =
    vkAcquireNextImageKHR(ctx_->device(), handle(), timeout, image_available, VK_NULL_HANDLE, &image_index);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) { return std::optional<std::uint32_t>{}; }
  if (result != VK_SUCCESS) { return fail(result, "vkAcquireNextImageKHR failed"); }
  return image_index;
}

auto owned::swapchain::present(std::uint32_t image_index,
  std::span<VkSemaphore const> wait_semaphores,
  present_options options) -> result<bool>
{
  if (ctx_ == nullptr || handle() == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "owned::swapchain::present on empty swapchain");
  }

  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.pNext = options.p_next;
  present_info.waitSemaphoreCount = static_cast<std::uint32_t>(wait_semaphores.size());
  present_info.pWaitSemaphores = wait_semaphores.empty() ? nullptr : wait_semaphores.data();
  present_info.swapchainCount = 1;
  auto *const swapchain_handle = handle();
  present_info.pSwapchains = &swapchain_handle;
  present_info.pImageIndices = &image_index;

  VkResult const result = vkQueuePresentKHR(ctx_->present_queue(), &present_info);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) { return false; }
  if (result != VK_SUCCESS) { return fail(result, "vkQueuePresentKHR failed"); }
  return true;
}

auto owned::swapchain::create_or_recreate(std::uint32_t width, std::uint32_t height) -> status
{
  if (width == 0 || height == 0) { return fail(errc::invalid_argument, "swapchain extent must be > 0"); }

  destroy_views();
  images_.clear();

  vkb::Swapchain const old_swapchain = impl_ ? impl_->swapchain : vkb::Swapchain{};
  auto builder =
    vkb::SwapchainBuilder{
      ctx_->physical_device(), ctx_->device(), surface_, ctx_->graphics_queue_family(), ctx_->present_queue_family()
    }
      .set_desired_format({ .format = preferred_format_, .colorSpace = preferred_color_space_ })
      .set_desired_present_mode(present_mode_)
      .set_desired_extent(width, height)
      .set_create_flags(static_cast<VkSwapchainCreateFlagBitsKHR>(create_flags_))
      .set_old_swapchain(old_swapchain);

  auto const built = builder.build();
  if (!built) { return fail(detail::make_error_from_vkb(built, "vk-bootstrap SwapchainBuilder")); }
  if (!impl_) { impl_ = std::make_unique<impl>(); }
  impl_->swapchain = built.value();
  if (old_swapchain.swapchain != VK_NULL_HANDLE) { vkb::destroy_swapchain(old_swapchain); }

  format_ = impl_->swapchain.image_format;
  extent_ = impl_->swapchain.extent;

  auto const got_images = impl_->swapchain.get_images();
  if (!got_images) { return fail(detail::make_error_from_vkb(got_images, "vk-bootstrap Swapchain::get_images")); }
  images_ = got_images.value();

  auto const got_views = impl_->swapchain.get_image_views();
  if (!got_views) { return fail(detail::make_error_from_vkb(got_views, "vk-bootstrap Swapchain::get_image_views")); }
  views_ = got_views.value();
  return {};
}

auto owned::swapchain::destroy_views() noexcept -> void
{
  if (ctx_ == nullptr || handle() == VK_NULL_HANDLE || views_.empty()) {
    views_.clear();
    return;
  }
  impl_->swapchain.destroy_image_views(views_);
  views_.clear();
}

auto owned::swapchain::destroy() noexcept -> void
{
  destroy_views();
  images_.clear();
  if (handle() != VK_NULL_HANDLE) {
    vkb::destroy_swapchain(impl_->swapchain);
    impl_.reset();
  }
  ctx_ = nullptr;
  surface_ = VK_NULL_HANDLE;
}

}// namespace vkexec
