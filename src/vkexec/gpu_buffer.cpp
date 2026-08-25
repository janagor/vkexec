#include <vkexec/gpu_buffer.hpp>

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <span>
#include <stdexcept>

namespace vkexec {
namespace {

  [[noreturn]] auto fail(char const *what) -> void { VKEXEC_THROW(std::runtime_error(what)); }

  auto usage_for(gpu_buffer_memory memory) -> VkBufferUsageFlags
  {
    (void)memory;
    // NOLINTBEGIN(hicpp-signed-bitwise)
    return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
           | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    // NOLINTEND(hicpp-signed-bitwise)
  }

  auto allocation_info_for(gpu_buffer_memory memory) -> VmaAllocationCreateInfo
  {
    VmaAllocationCreateInfo aci{};
    switch (memory) {
    case gpu_buffer_memory::host_visible:
      aci.usage = VMA_MEMORY_USAGE_AUTO;
      // NOLINTBEGIN(hicpp-signed-bitwise)
      aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
      aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      // NOLINTEND(hicpp-signed-bitwise)
      break;
    case gpu_buffer_memory::device_local:
      aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
      break;
    }
    return aci;
  }

}// namespace

auto gpu_buffer::create(context &ctx, VkDeviceSize size, gpu_buffer_memory memory) -> gpu_buffer
{
  if (size == 0) { VKEXEC_THROW(std::invalid_argument("vkexec::gpu_buffer size must be > 0")); }
  if (ctx.allocator() == VK_NULL_HANDLE) { fail("vkexec::gpu_buffer requires a VMA allocator"); }

  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = size;
  bci.usage = usage_for(memory);
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo const aci = allocation_info_for(memory);

  VkBuffer buffer_handle{ VK_NULL_HANDLE };
  VmaAllocation allocation{ VK_NULL_HANDLE };
  VmaAllocationInfo ainfo{};
  if (vmaCreateBuffer(ctx.allocator(), &bci, &aci, &buffer_handle, &allocation, &ainfo) != VK_SUCCESS) {
    fail("vmaCreateBuffer failed");
  }

  void *mapped_ptr = nullptr;
  if (memory == gpu_buffer_memory::host_visible) {
    mapped_ptr = ainfo.pMappedData;
    if (mapped_ptr == nullptr) {
      vmaDestroyBuffer(ctx.allocator(), buffer_handle, allocation);
      fail("vmaCreateBuffer did not map host-visible memory");
    }
  }

  return gpu_buffer{ &ctx, buffer_handle, allocation, mapped_ptr, size, memory };
}

gpu_buffer::gpu_buffer(context *ctx,
  VkBuffer buffer,
  VmaAllocation allocation,
  void *mapped,
  VkDeviceSize size,
  gpu_buffer_memory memory) noexcept
  : ctx_(ctx), buffer_(buffer), allocation_(allocation), mapped_(mapped), size_(size), memory_(memory)
{}

gpu_buffer::~gpu_buffer() { destroy(); }

gpu_buffer::gpu_buffer(gpu_buffer &&other) noexcept
  : ctx_(other.ctx_), buffer_(other.buffer_), allocation_(other.allocation_), mapped_(other.mapped_),
    size_(other.size_), memory_(other.memory_)
{
  other.ctx_ = nullptr;
  other.buffer_ = VK_NULL_HANDLE;
  other.allocation_ = VK_NULL_HANDLE;
  other.mapped_ = nullptr;
  other.size_ = 0;
}

auto gpu_buffer::operator=(gpu_buffer &&other) noexcept -> gpu_buffer &
{
  if (this == &other) { return *this; }
  destroy();
  ctx_ = other.ctx_;
  buffer_ = other.buffer_;
  allocation_ = other.allocation_;
  mapped_ = other.mapped_;
  size_ = other.size_;
  memory_ = other.memory_;
  other.ctx_ = nullptr;
  other.buffer_ = VK_NULL_HANDLE;
  other.allocation_ = VK_NULL_HANDLE;
  other.mapped_ = nullptr;
  other.size_ = 0;
  return *this;
}

auto gpu_buffer::mapped() const noexcept -> std::span<std::byte>
{
  if (mapped_ == nullptr) { return {}; }
  return { static_cast<std::byte *>(mapped_), static_cast<std::size_t>(size_) };
}

auto gpu_buffer::destroy() noexcept -> void
{
  if (ctx_ != nullptr && ctx_->allocator() != VK_NULL_HANDLE && buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(ctx_->allocator(), buffer_, allocation_);
  }
  ctx_ = nullptr;
  buffer_ = VK_NULL_HANDLE;
  allocation_ = VK_NULL_HANDLE;
  mapped_ = nullptr;
  size_ = 0;
}

}// namespace vkexec
