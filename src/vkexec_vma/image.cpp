#include <vkexec_vma/image.hpp>

#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/result.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

namespace vkexec {
namespace {

  auto resolve_format(vma::image_create_info const &info) -> result<VkFormat>
  {
    // Defaults bias toward HDR storage / 32-bit depth when the caller leaves format unset.
    if (info.format != VK_FORMAT_UNDEFINED) { return info.format; }
    switch (info.usage) {
    case vma::image_usage::color_storage:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
    case vma::image_usage::depth:
      return VK_FORMAT_D32_SFLOAT;
    }
    return fail(errc::invalid_argument, "unknown image_usage");
  }

  auto usage_flags(vma::image_usage usage) -> result<VkImageUsageFlags>
  {
    switch (usage) {
    case vma::image_usage::color_storage:
      // NOLINTBEGIN(hicpp-signed-bitwise)
      return VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
             | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      // NOLINTEND(hicpp-signed-bitwise)
    case vma::image_usage::depth:
      return VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }
    return fail(errc::invalid_argument, "unknown image_usage");
  }

}// namespace

auto detail::make_image_factory::operator()() const -> result<::vkexec::vma::image>
{
  if (info.width == 0 || info.height == 0) { return fail(errc::invalid_argument, "vkexec::image extent must be > 0"); }
  if (allocator == nullptr || allocator->native_handle() == VK_NULL_HANDLE) {
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
    vmaCreateImage(allocator->native_handle(), &image_info, &alloc_info, &image_handle, &allocation, nullptr);
  if (create_result != VK_SUCCESS) { return fail(create_result, "vmaCreateImage failed"); }

  return ::vkexec::vma::image{ allocator->native_handle(),
    image_handle,
    allocation,
    vk_format,
    VkExtent2D{ .width = info.width, .height = info.height },
    info.usage,
    usg_flags };
}

auto detail::make_generic_image_factory::operator()() const -> result<::vkexec::vma::image>
{
  if (allocator == nullptr || allocator->native_handle() == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "allocate_image requires a live allocator");
  }
  if (info.extent.width == 0 || info.extent.height == 0 || info.extent.depth != 1 || info.format == VK_FORMAT_UNDEFINED
      || info.usage == 0) {
    return fail(errc::invalid_argument, "allocate_image requires a 2D extent, format, and usage");
  }

  // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
  VkImageCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  create_info.imageType = VK_IMAGE_TYPE_2D;
  create_info.extent = info.extent;
  create_info.mipLevels = 1;
  create_info.arrayLayers = 1;
  create_info.format = info.format;
  create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  create_info.usage = info.usage;
  create_info.samples = VK_SAMPLE_COUNT_1_BIT;
  create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo allocation_info{};
  allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  VkImage image_handle{ VK_NULL_HANDLE };
  VmaAllocation allocation{ VK_NULL_HANDLE };
  VkResult const created =
    vmaCreateImage(allocator->native_handle(), &create_info, &allocation_info, &image_handle, &allocation, nullptr);
  if (created != VK_SUCCESS) { return fail(created, "vmaCreateImage failed"); }
  auto const usage = (info.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0 ? vma::image_usage::depth
                                                                                     : vma::image_usage::color_storage;
  return vma::image{ allocator->native_handle(),
    image_handle,
    allocation,
    info.format,
    VkExtent2D{ .width = info.extent.width, .height = info.extent.height },
    usage,
    info.usage };
}

vma::image::image(VmaAllocator allocator_handle,
  VkImage image_handle,
  VmaAllocation allocation,
  VkFormat format,
  VkExtent2D extent,
  vma::image_usage usage,
  VkImageUsageFlags usage_flags) noexcept
  : allocator_(allocator_handle), image_(image_handle), allocation_(allocation), format_(format), extent_(extent),
    usage_(usage), usage_flags_(usage_flags)
{}

vma::image::~image() { destroy(); }

vma::image::image(image &&other) noexcept
  : allocator_(other.allocator_), image_(other.image_), allocation_(other.allocation_), format_(other.format_),
    extent_(other.extent_), usage_(other.usage_), usage_flags_(other.usage_flags_)
{
  other.allocator_ = nullptr;
  other.image_ = VK_NULL_HANDLE;
  other.allocation_ = VK_NULL_HANDLE;
  other.usage_flags_ = 0;
}

auto vma::image::operator=(image &&other) noexcept -> image &
{
  if (this == &other) { return *this; }
  destroy();
  allocator_ = other.allocator_;
  image_ = other.image_;
  allocation_ = other.allocation_;
  format_ = other.format_;
  extent_ = other.extent_;
  usage_ = other.usage_;
  usage_flags_ = other.usage_flags_;
  other.allocator_ = nullptr;
  other.image_ = VK_NULL_HANDLE;
  other.allocation_ = VK_NULL_HANDLE;
  other.usage_flags_ = 0;
  return *this;
}

auto vma::image::destroy() noexcept -> void
{
  if (allocator_ != VK_NULL_HANDLE && image_ != VK_NULL_HANDLE) { vmaDestroyImage(allocator_, image_, allocation_); }
  allocator_ = VK_NULL_HANDLE;
  image_ = VK_NULL_HANDLE;
  allocation_ = VK_NULL_HANDLE;
  usage_flags_ = 0;
}

}// namespace vkexec
