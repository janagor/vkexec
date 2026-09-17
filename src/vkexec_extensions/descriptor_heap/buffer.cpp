#include <vkexec_extensions/descriptor_heap/buffer.hpp>

#include <vkexec/context.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/result.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <span>

namespace vkexec {
namespace {

  constexpr VkDeviceSize k_heap_device_address_alignment = 4096;

}// namespace

auto descriptor_heap_buffer::create(context &ctx, VkDeviceSize size) -> detail::sync_sender_fn<descriptor_heap_buffer>
{
  return detail::make_sync_sender_fn<descriptor_heap_buffer>([&ctx, size]() -> result<descriptor_heap_buffer> {
    if (size == 0) { return fail(errc::invalid_argument, "descriptor_heap_buffer size must be > 0"); }
    if (ctx.allocator() == VK_NULL_HANDLE) {
      return fail(errc::invalid_argument, "descriptor_heap_buffer requires a VMA allocator");
    }
    if (ctx.procs().get_buffer_device_address == nullptr) {
      return fail(errc::unsupported, "descriptor_heap_buffer requires vkGetBufferDeviceAddress on the device");
    }

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    // NOLINTBEGIN(hicpp-signed-bitwise)
    bci.usage = VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    // NOLINTEND(hicpp-signed-bitwise)
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    // NOLINTBEGIN(hicpp-signed-bitwise)
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
                | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    // NOLINTEND(hicpp-signed-bitwise)

    VkBuffer buffer_handle{ VK_NULL_HANDLE };
    VmaAllocation allocation{ VK_NULL_HANDLE };
    VmaAllocationInfo ainfo{};
    VkResult const create_result = vmaCreateBufferWithAlignment(
      ctx.allocator(), &bci, &aci, k_heap_device_address_alignment, &buffer_handle, &allocation, &ainfo);
    if (create_result != VK_SUCCESS) { return fail(create_result, "vmaCreateBufferWithAlignment failed"); }

    void *const mapped_ptr = ainfo.pMappedData;
    if (mapped_ptr == nullptr) {
      vmaDestroyBuffer(ctx.allocator(), buffer_handle, allocation);
      return fail(errc::unsupported, "descriptor_heap_buffer did not map host-visible memory");
    }

    VkMemoryPropertyFlags memory_flags = 0;
    vmaGetAllocationMemoryProperties(ctx.allocator(), allocation, &memory_flags);
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    bool const host_coherent = (memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

    return descriptor_heap_buffer{ &ctx, buffer_handle, allocation, mapped_ptr, size, host_coherent };
  });
}

descriptor_heap_buffer::descriptor_heap_buffer(context *ctx,
  VkBuffer buffer,
  VmaAllocation allocation,
  void *mapped,
  VkDeviceSize size,
  bool host_coherent) noexcept
  : ctx_(ctx), buffer_(buffer), allocation_(allocation), mapped_(mapped), size_(size), host_coherent_(host_coherent)
{}

descriptor_heap_buffer::~descriptor_heap_buffer() { destroy(); }

descriptor_heap_buffer::descriptor_heap_buffer(descriptor_heap_buffer &&other) noexcept
  : ctx_(other.ctx_), buffer_(other.buffer_), allocation_(other.allocation_), mapped_(other.mapped_), size_(other.size_),
    host_coherent_(other.host_coherent_)
{
  other.ctx_ = nullptr;
  other.buffer_ = VK_NULL_HANDLE;
  other.allocation_ = VK_NULL_HANDLE;
  other.mapped_ = nullptr;
  other.size_ = 0;
  other.host_coherent_ = true;
}

auto descriptor_heap_buffer::operator=(descriptor_heap_buffer &&other) noexcept -> descriptor_heap_buffer &
{
  if (this == &other) { return *this; }
  destroy();
  ctx_ = other.ctx_;
  buffer_ = other.buffer_;
  allocation_ = other.allocation_;
  mapped_ = other.mapped_;
  size_ = other.size_;
  host_coherent_ = other.host_coherent_;
  other.ctx_ = nullptr;
  other.buffer_ = VK_NULL_HANDLE;
  other.allocation_ = VK_NULL_HANDLE;
  other.mapped_ = nullptr;
  other.size_ = 0;
  other.host_coherent_ = true;
  return *this;
}

auto descriptor_heap_buffer::mapped() const noexcept -> std::span<std::byte>
{
  if (mapped_ == nullptr) { return {}; }
  return { static_cast<std::byte *>(mapped_), static_cast<std::size_t>(size_) };
}

auto descriptor_heap_buffer::device_address() const -> result<VkDeviceAddress>
{
  if (ctx_ == nullptr || ctx_->procs().get_buffer_device_address == nullptr) {
    return fail(errc::unsupported, "vkGetBufferDeviceAddress is unavailable");
  }
  VkBufferDeviceAddressInfo info{};
  info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  info.buffer = buffer_;
  return ctx_->procs().get_buffer_device_address(ctx_->device(), &info);
}

auto descriptor_heap_buffer::flush() const -> status
{
  if (host_coherent_) { return {}; }
  if (ctx_ == nullptr || ctx_->allocator() == VK_NULL_HANDLE || allocation_ == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "descriptor_heap_buffer::flush requires a live allocation");
  }
  VkResult const flush_result = vmaFlushAllocation(ctx_->allocator(), allocation_, 0, VK_WHOLE_SIZE);
  if (flush_result != VK_SUCCESS) { return fail(flush_result, "vmaFlushAllocation failed"); }
  return {};
}

auto descriptor_heap_buffer::destroy() noexcept -> void
{
  if (ctx_ != nullptr && ctx_->allocator() != VK_NULL_HANDLE && buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(ctx_->allocator(), buffer_, allocation_);
  }
  ctx_ = nullptr;
  buffer_ = VK_NULL_HANDLE;
  allocation_ = VK_NULL_HANDLE;
  mapped_ = nullptr;
  size_ = 0;
  host_coherent_ = true;
}

}// namespace vkexec
