#include <vkexec/image.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

namespace vkexec {
namespace {

  auto resolve_format(image_create_info const &info) -> result<VkFormat>
  {
    // Defaults bias toward HDR storage / 32-bit depth when the caller leaves format unset.
    if (info.format != VK_FORMAT_UNDEFINED) { return info.format; }
    switch (info.usage) {
    case image_usage::color_storage:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
    case image_usage::depth:
      return VK_FORMAT_D32_SFLOAT;
    }
    return fail(errc::invalid_argument, "unknown image_usage");
  }

  auto usage_flags(image_usage usage) -> result<VkImageUsageFlags>
  {
    switch (usage) {
    case image_usage::color_storage:
      // NOLINTBEGIN(hicpp-signed-bitwise)
      return VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
             | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      // NOLINTEND(hicpp-signed-bitwise)
    case image_usage::depth:
      return VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }
    return fail(errc::invalid_argument, "unknown image_usage");
  }

}// namespace

auto image::create(context &ctx, image_create_info info) -> sender<image>
{
  return make_sender<image>([&ctx, info]() -> result<image> {
    if (info.width == 0 || info.height == 0) {
      return fail(errc::invalid_argument, "vkexec::image extent must be > 0");
    }
    if (ctx.allocator() == VK_NULL_HANDLE) {
      return fail(errc::invalid_argument, "vkexec::image requires a VMA allocator");
    }

    VKEXEC_TRY_ASSIGN(vk_format, resolve_format(info));
    VKEXEC_TRY_ASSIGN(usg_flags, usage_flags(info.usage));

    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent = { .width = info.width, .height = info.height, .depth = 1 };
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = vk_format;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = usg_flags;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkImage image_handle{ VK_NULL_HANDLE };
    VmaAllocation allocation{ VK_NULL_HANDLE };
    VkResult const create_result =
      vmaCreateImage(ctx.allocator(), &image_info, &alloc_info, &image_handle, &allocation, nullptr);
    if (create_result != VK_SUCCESS) { return fail(create_result, "vmaCreateImage failed"); }

    return image{
      &ctx, image_handle, allocation, vk_format, VkExtent2D{ .width = info.width, .height = info.height }, info.usage
    };
  });
}

image::image(context *ctx,
  VkImage image,
  VmaAllocation allocation,
  VkFormat format,
  VkExtent2D extent,
  image_usage usage) noexcept
  : ctx_(ctx), image_(image), allocation_(allocation), format_(format), extent_(extent), usage_(usage)
{}

image::~image() { destroy(); }

image::image(image &&other) noexcept
  : ctx_(other.ctx_), image_(other.image_), allocation_(other.allocation_), format_(other.format_),
    extent_(other.extent_), usage_(other.usage_)
{
  other.ctx_ = nullptr;
  other.image_ = VK_NULL_HANDLE;
  other.allocation_ = VK_NULL_HANDLE;
}

auto image::operator=(image &&other) noexcept -> image &
{
  if (this == &other) { return *this; }
  destroy();
  ctx_ = other.ctx_;
  image_ = other.image_;
  allocation_ = other.allocation_;
  format_ = other.format_;
  extent_ = other.extent_;
  usage_ = other.usage_;
  other.ctx_ = nullptr;
  other.image_ = VK_NULL_HANDLE;
  other.allocation_ = VK_NULL_HANDLE;
  return *this;
}

auto image::destroy() noexcept -> void
{
  if (ctx_ != nullptr && ctx_->allocator() != VK_NULL_HANDLE && image_ != VK_NULL_HANDLE) {
    vmaDestroyImage(ctx_->allocator(), image_, allocation_);
  }
  ctx_ = nullptr;
  image_ = VK_NULL_HANDLE;
  allocation_ = VK_NULL_HANDLE;
}

}// namespace vkexec
