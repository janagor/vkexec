#include <vkexec/image_view.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/image.hpp>

#include <vulkan/vulkan_core.h>

namespace vkexec {
namespace {

  auto aspect_for(image_usage usage) -> VkImageAspectFlags
  { return usage == image_usage::depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT; }

}// namespace

auto image_view::create(context &ctx, image const &img) -> result<image_view>
{
  if (ctx.device() == VK_NULL_HANDLE) {
    return std::unexpected(make_error(errc::invalid_argument, "image_view requires a VkDevice"));
  }
  if (img.handle() == VK_NULL_HANDLE) {
    return std::unexpected(make_error(errc::invalid_argument, "image_view requires a valid image"));
  }

  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = img.handle();
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = img.format();
  view_info.subresourceRange.aspectMask = aspect_for(img.usage());
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = 1;

  VkImageView view{ VK_NULL_HANDLE };
  VkResult const create_result = vkCreateImageView(ctx.device(), &view_info, nullptr, &view);
  if (create_result != VK_SUCCESS) {
    return std::unexpected(make_vk_error(create_result, "vkCreateImageView failed"));
  }
  return image_view{ &ctx, view };
}

image_view::image_view(context *ctx, VkImageView view) noexcept : ctx_(ctx), view_(view) {}

image_view::~image_view() { destroy(); }

image_view::image_view(image_view &&other) noexcept : ctx_(other.ctx_), view_(other.view_)
{
  other.ctx_ = nullptr;
  other.view_ = VK_NULL_HANDLE;
}

auto image_view::operator=(image_view &&other) noexcept -> image_view &
{
  if (this == &other) { return *this; }
  destroy();
  ctx_ = other.ctx_;
  view_ = other.view_;
  other.ctx_ = nullptr;
  other.view_ = VK_NULL_HANDLE;
  return *this;
}

auto image_view::destroy() noexcept -> void
{
  if (ctx_ != nullptr && ctx_->device() != VK_NULL_HANDLE && view_ != VK_NULL_HANDLE) {
    vkDestroyImageView(ctx_->device(), view_, nullptr);
  }
  ctx_ = nullptr;
  view_ = VK_NULL_HANDLE;
}

}// namespace vkexec
