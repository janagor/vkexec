#include <vkexec/image.hpp>

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <stdexcept>

namespace vkexec {
namespace {

  [[noreturn]] auto fail(char const *what) -> void { VKEXEC_THROW(std::runtime_error(what)); }

  auto resolve_format(image_create_info const &info) -> VkFormat
  {
    if (info.format != VK_FORMAT_UNDEFINED) { return info.format; }
    switch (info.usage) {
    case image_usage::color_storage:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
    case image_usage::depth:
      return VK_FORMAT_D32_SFLOAT;
    }
    fail("unknown image_usage");
  }

  auto usage_flags(image_usage usage) -> VkImageUsageFlags
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
    fail("unknown image_usage");
  }

}// namespace

auto image::create(context &ctx, image_create_info info) -> image
{
  if (info.width == 0 || info.height == 0) { VKEXEC_THROW(std::invalid_argument("vkexec::image extent must be > 0")); }
  if (ctx.allocator() == VK_NULL_HANDLE) { fail("vkexec::image requires a VMA allocator"); }

  VkFormat const format = resolve_format(info);

  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.extent = { info.width, info.height, 1 };
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.format = format;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_info.usage = usage_flags(info.usage);
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  VkImage image_handle{ VK_NULL_HANDLE };
  VmaAllocation allocation{ VK_NULL_HANDLE };
  if (vmaCreateImage(ctx.allocator(), &image_info, &alloc_info, &image_handle, &allocation, nullptr) != VK_SUCCESS) {
    fail("vmaCreateImage failed");
  }

  return image{ &ctx, image_handle, allocation, format, VkExtent2D{ info.width, info.height }, info.usage };
}

image::image(context *ctx,
  VkImage image_handle,
  VmaAllocation allocation,
  VkFormat format,
  VkExtent2D extent,
  image_usage usage) noexcept
  : ctx_(ctx), image_(image_handle), allocation_(allocation), format_(format), extent_(extent), usage_(usage)
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
