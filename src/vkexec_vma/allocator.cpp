#include <vkexec_vma/allocator.hpp>

#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan_core.h>

namespace vkexec::vma {

auto detail::make_allocator_factory::operator()() const -> result<allocator>
{
  if (ctx == nullptr || ctx->instance() == VK_NULL_HANDLE || ctx->physical_device() == VK_NULL_HANDLE
      || ctx->device() == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "vma::make_allocator requires an initialized Vulkan context");
  }

  VmaAllocatorCreateInfo info{};
  info.instance = ctx->instance();
  info.physicalDevice = ctx->physical_device();
  info.device = ctx->device();
  info.vulkanApiVersion = ctx->api_version();
  if (ctx->procs().get_buffer_device_address != nullptr) {
    info.flags |= static_cast<decltype(info.flags)>(VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT);
  }

  VmaAllocator handle{ VK_NULL_HANDLE };
  if (VkResult const created = vmaCreateAllocator(&info, &handle); created != VK_SUCCESS) {
    return fail(created, "vmaCreateAllocator failed");
  }
  return allocator{ ctx, handle, true };
}

auto detail::adopt_allocator_factory::operator()() const -> result<allocator>
{
  if (ctx == nullptr || handle == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "vma::adopt_allocator requires a context and VmaAllocator");
  }
  return allocator{ ctx, handle, false };
}

allocator::~allocator() { destroy(); }

allocator::allocator(allocator &&other) noexcept : ctx_(other.ctx_), handle_(other.handle_), owns_(other.owns_)
{
  other.ctx_ = nullptr;
  other.handle_ = VK_NULL_HANDLE;
  other.owns_ = false;
}

auto allocator::operator=(allocator &&other) noexcept -> allocator &
{
  if (this == &other) { return *this; }
  destroy();
  ctx_ = other.ctx_;
  handle_ = other.handle_;
  owns_ = other.owns_;
  other.ctx_ = nullptr;
  other.handle_ = VK_NULL_HANDLE;
  other.owns_ = false;
  return *this;
}

auto allocator::destroy() noexcept -> void
{
  if (owns_ && handle_ != VK_NULL_HANDLE) { vmaDestroyAllocator(handle_); }
  ctx_ = nullptr;
  handle_ = VK_NULL_HANDLE;
  owns_ = false;
}

}// namespace vkexec::vma
