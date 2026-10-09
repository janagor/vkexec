#include <vkexec_vma/gpu_buffer.hpp>

#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/result.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <span>

namespace vkexec {
namespace {

  auto usage_for(vma::gpu_buffer_memory memory, bool shader_device_address) -> result<VkBufferUsageFlags>
  {
    switch (memory) {
    case vma::gpu_buffer_memory::host_visible:
    case vma::gpu_buffer_memory::device_local: {
      // Storage + transfer + indirect covers typical compute/hybrid upload paths.
      VkBufferUsageFlags usage = static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
                                 | static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                                 | static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
                                 | static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
      if (shader_device_address) {
        usage |= static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
      }
      return usage;
    }
    case vma::gpu_buffer_memory::staging:
      // Upload and readback both need copy endpoints on the staging buffer.
      return static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
             | static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }
    return fail(errc::invalid_argument, "unknown gpu_buffer_memory");
  }

  auto allocation_info_for(vma::gpu_buffer_memory memory) -> VmaAllocationCreateInfo
  {
    VmaAllocationCreateInfo aci{};
    switch (memory) {
    case vma::gpu_buffer_memory::host_visible:
      aci.usage = VMA_MEMORY_USAGE_AUTO;
      aci.flags = static_cast<VmaAllocationCreateFlags>(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT)
                  | static_cast<VmaAllocationCreateFlags>(VMA_ALLOCATION_CREATE_MAPPED_BIT);
      aci.requiredFlags = static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                          | static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      break;
    case vma::gpu_buffer_memory::device_local:
      aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
      break;
    case vma::gpu_buffer_memory::staging:
      // Readback-friendly mapping: random host access, prefer coherent if available.
      aci.usage = VMA_MEMORY_USAGE_AUTO;
      aci.flags = static_cast<VmaAllocationCreateFlags>(VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)
                  | static_cast<VmaAllocationCreateFlags>(VMA_ALLOCATION_CREATE_MAPPED_BIT);
      aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
      aci.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      break;
    }
    return aci;
  }

}// namespace

auto detail::make_gpu_buffer_factory::operator()() const -> result<::vkexec::vma::gpu_buffer>
{
  if (info.size == 0) { return fail(errc::invalid_argument, "vkexec::gpu_buffer size must be > 0"); }
  if (allocator == nullptr || allocator->native_handle() == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "vkexec::gpu_buffer requires a VMA allocator");
  }

  bool const want_device_address = info.shader_device_address;
  if (want_device_address && !allocator->ctx().capabilities().buffer_device_address) {
    return fail(
      errc::unsupported, "vkexec::gpu_buffer shader device address requested but bufferDeviceAddress is not enabled");
  }

  VKEXEC_TRY_ASSIGN(usage, usage_for(info.memory, want_device_address));
  usage |= info.usage;

  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = info.size;
  bci.usage = usage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo const aci = allocation_info_for(info.memory);

  VkBuffer buffer_handle{ VK_NULL_HANDLE };
  VmaAllocation allocation{ VK_NULL_HANDLE };
  VmaAllocationInfo ainfo{};
  VkResult const create_result =
    info.alignment == 0
      ? vmaCreateBuffer(allocator->native_handle(), &bci, &aci, &buffer_handle, &allocation, &ainfo)
      : vmaCreateBufferWithAlignment(
          allocator->native_handle(), &bci, &aci, info.alignment, &buffer_handle, &allocation, &ainfo);
  if (create_result != VK_SUCCESS) { return fail(create_result, "vmaCreateBuffer failed"); }

  void *mapped_ptr = nullptr;
  if (info.memory == vma::gpu_buffer_memory::host_visible || info.memory == vma::gpu_buffer_memory::staging) {
    mapped_ptr = ainfo.pMappedData;
    if (mapped_ptr == nullptr) {
      vmaDestroyBuffer(allocator->native_handle(), buffer_handle, allocation);
      return fail(errc::unsupported, "vmaCreateBuffer did not map host-visible memory");
    }
  }

  return ::vkexec::vma::gpu_buffer{ &allocator->ctx(),
    allocator->native_handle(),
    buffer_handle,
    allocation,
    mapped_ptr,
    info.size,
    info.memory,
    want_device_address };
}

vma::gpu_buffer::gpu_buffer(context *ctx,
  VmaAllocator allocator,
  VkBuffer buffer,
  VmaAllocation allocation,
  void *mapped,
  VkDeviceSize size,
  gpu_buffer_memory memory,
  bool shader_device_address) noexcept
  : ctx_(ctx), allocator_(allocator), buffer_(buffer), allocation_(allocation), mapped_(mapped), size_(size),
    memory_(memory), shader_device_address_(shader_device_address)
{}

vma::gpu_buffer::~gpu_buffer() { destroy(); }

vma::gpu_buffer::gpu_buffer(gpu_buffer &&other) noexcept
  : ctx_(other.ctx_), allocator_(other.allocator_), buffer_(other.buffer_), allocation_(other.allocation_),
    mapped_(other.mapped_), size_(other.size_), memory_(other.memory_),
    shader_device_address_(other.shader_device_address_)
{
  other.ctx_ = nullptr;
  other.allocator_ = VK_NULL_HANDLE;
  other.buffer_ = VK_NULL_HANDLE;
  other.allocation_ = VK_NULL_HANDLE;
  other.mapped_ = nullptr;
  other.size_ = 0;
  other.shader_device_address_ = false;
}

auto vma::gpu_buffer::operator=(gpu_buffer &&other) noexcept -> gpu_buffer &
{
  if (this == &other) { return *this; }
  destroy();
  ctx_ = other.ctx_;
  allocator_ = other.allocator_;
  buffer_ = other.buffer_;
  allocation_ = other.allocation_;
  mapped_ = other.mapped_;
  size_ = other.size_;
  memory_ = other.memory_;
  shader_device_address_ = other.shader_device_address_;
  other.ctx_ = nullptr;
  other.allocator_ = VK_NULL_HANDLE;
  other.buffer_ = VK_NULL_HANDLE;
  other.allocation_ = VK_NULL_HANDLE;
  other.mapped_ = nullptr;
  other.size_ = 0;
  other.shader_device_address_ = false;
  return *this;
}

auto vma::gpu_buffer::mapped() noexcept -> std::span<std::byte>
{
  if (mapped_ == nullptr) { return {}; }
  return { static_cast<std::byte *>(mapped_), size_ };
}

auto vma::gpu_buffer::mapped() const noexcept -> std::span<std::byte const>
{
  if (mapped_ == nullptr) { return {}; }
  return { static_cast<std::byte const *>(mapped_), size_ };
}

auto vma::gpu_buffer::flush() const -> status
{
  if (allocator_ == VK_NULL_HANDLE || allocation_ == VK_NULL_HANDLE || mapped_ == nullptr) {
    return fail(errc::invalid_argument, "flush requires a mapped VMA buffer");
  }
  VkResult const flushed = vmaFlushAllocation(allocator_, allocation_, 0, VK_WHOLE_SIZE);
  if (flushed != VK_SUCCESS) { return fail(flushed, "vmaFlushAllocation failed"); }
  return {};
}

auto vma::gpu_buffer::invalidate() const -> status
{
  if (allocator_ == VK_NULL_HANDLE || allocation_ == VK_NULL_HANDLE || mapped_ == nullptr) {
    return fail(errc::invalid_argument, "invalidate requires a mapped VMA buffer");
  }
  VkResult const invalidated = vmaInvalidateAllocation(allocator_, allocation_, 0, VK_WHOLE_SIZE);
  if (invalidated != VK_SUCCESS) { return fail(invalidated, "vmaInvalidateAllocation failed"); }
  return {};
}

auto vma::gpu_buffer::device_address() const -> result<VkDeviceAddress>
{
  if (!shader_device_address_) {
    return fail(errc::invalid_argument, "vkexec::gpu_buffer was not created with shader_device_address");
  }
  if (ctx_ == nullptr || ctx_->procs().get_buffer_device_address == nullptr) {
    return fail(errc::unsupported, "vkGetBufferDeviceAddress is unavailable");
  }
  VkBufferDeviceAddressInfo info{};
  info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  info.buffer = buffer_;
  return ctx_->procs().get_buffer_device_address(ctx_->device(), &info);
}

auto vma::gpu_buffer::destroy() noexcept -> void
{
  if (allocator_ != VK_NULL_HANDLE && buffer_ != VK_NULL_HANDLE) { vmaDestroyBuffer(allocator_, buffer_, allocation_); }
  ctx_ = nullptr;
  allocator_ = VK_NULL_HANDLE;
  buffer_ = VK_NULL_HANDLE;
  allocation_ = VK_NULL_HANDLE;
  mapped_ = nullptr;
  size_ = 0;
  shader_device_address_ = false;
}

}// namespace vkexec
