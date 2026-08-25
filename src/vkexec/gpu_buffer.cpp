#include <vkexec/gpu_buffer.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error_helpers.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <span>

namespace vkexec {
namespace {

  auto usage_for(gpu_buffer_memory memory, bool shader_device_address) -> result<VkBufferUsageFlags>
  {
    switch (memory) {
    case gpu_buffer_memory::host_visible:
    case gpu_buffer_memory::device_local: {
      // NOLINTBEGIN(hicpp-signed-bitwise)
      VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                 | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
      if (shader_device_address) { usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT; }
      return usage;
      // NOLINTEND(hicpp-signed-bitwise)
    }
    case gpu_buffer_memory::staging:
      return VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    case gpu_buffer_memory::descriptor_heap:
      // NOLINTBEGIN(hicpp-signed-bitwise)
      return VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      // NOLINTEND(hicpp-signed-bitwise)
    }
    return std::unexpected(make_error(errc::invalid_argument, "unknown gpu_buffer_memory"));
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
    case gpu_buffer_memory::staging:
      aci.usage = VMA_MEMORY_USAGE_AUTO;
      // NOLINTBEGIN(hicpp-signed-bitwise)
      aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
      // NOLINTEND(hicpp-signed-bitwise)
      aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
      aci.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      break;
    case gpu_buffer_memory::descriptor_heap:
      aci.usage = VMA_MEMORY_USAGE_AUTO;
      // NOLINTBEGIN(hicpp-signed-bitwise)
      // Persistently mapped heaps are written at arbitrary slot offsets; dedicated +
      // 4 KiB alignment matches ANV bindless heap addressing.
      aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
                  | VMA_ALLOCATION_CREATE_MAPPED_BIT;
      // NOLINTEND(hicpp-signed-bitwise)
      aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
      aci.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      break;
    }
    return aci;
  }

  constexpr VkDeviceSize k_heap_device_address_alignment = 4096;

}// namespace

auto gpu_buffer::create(context &ctx, gpu_buffer_create_info info) -> result<gpu_buffer>
{
  if (info.size == 0) {
    return std::unexpected(make_error(errc::invalid_argument, "vkexec::gpu_buffer size must be > 0"));
  }
  if (ctx.allocator() == VK_NULL_HANDLE) {
    return std::unexpected(make_error(errc::invalid_argument, "vkexec::gpu_buffer requires a VMA allocator"));
  }

  bool const want_device_address =
    info.shader_device_address || info.memory == gpu_buffer_memory::descriptor_heap;
  if (want_device_address && ctx.procs().get_buffer_device_address == nullptr) {
    return std::unexpected(make_error(errc::unsupported,
      "vkexec::gpu_buffer shader device address requested but vkGetBufferDeviceAddress is unavailable"));
  }

  auto const usage_result = usage_for(info.memory, want_device_address);
  if (!usage_result) { return std::unexpected(usage_result.error()); }

  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = info.size;
  bci.usage = *usage_result;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo const aci = allocation_info_for(info.memory);

  VkBuffer buffer_handle{ VK_NULL_HANDLE };
  VmaAllocation allocation{ VK_NULL_HANDLE };
  VmaAllocationInfo ainfo{};
  VkResult const create_result = info.memory == gpu_buffer_memory::descriptor_heap
                                   ? vmaCreateBufferWithAlignment(ctx.allocator(),
                                       &bci,
                                       &aci,
                                       k_heap_device_address_alignment,
                                       &buffer_handle,
                                       &allocation,
                                       &ainfo)
                                   : vmaCreateBuffer(ctx.allocator(), &bci, &aci, &buffer_handle, &allocation, &ainfo);
  if (create_result != VK_SUCCESS) {
    return std::unexpected(make_vk_error(create_result, "vmaCreateBuffer failed"));
  }

  void *mapped_ptr = nullptr;
  if (info.memory == gpu_buffer_memory::host_visible || info.memory == gpu_buffer_memory::staging
      || info.memory == gpu_buffer_memory::descriptor_heap) {
    mapped_ptr = ainfo.pMappedData;
    if (mapped_ptr == nullptr) {
      vmaDestroyBuffer(ctx.allocator(), buffer_handle, allocation);
      return std::unexpected(
        make_error(errc::unsupported, "vmaCreateBuffer did not map host-visible memory"));
    }
  }

  return gpu_buffer{ &ctx, buffer_handle, allocation, mapped_ptr, info.size, info.memory, want_device_address };
}

gpu_buffer::gpu_buffer(context *ctx,
  VkBuffer buffer,
  VmaAllocation allocation,
  void *mapped,
  VkDeviceSize size,
  gpu_buffer_memory memory,
  bool shader_device_address) noexcept
  : ctx_(ctx), buffer_(buffer), allocation_(allocation), mapped_(mapped), size_(size), memory_(memory),
    shader_device_address_(shader_device_address)
{}

gpu_buffer::~gpu_buffer() { destroy(); }

gpu_buffer::gpu_buffer(gpu_buffer &&other) noexcept
  : ctx_(other.ctx_), buffer_(other.buffer_), allocation_(other.allocation_), mapped_(other.mapped_),
    size_(other.size_), memory_(other.memory_), shader_device_address_(other.shader_device_address_)
{
  other.ctx_ = nullptr;
  other.buffer_ = VK_NULL_HANDLE;
  other.allocation_ = VK_NULL_HANDLE;
  other.mapped_ = nullptr;
  other.size_ = 0;
  other.shader_device_address_ = false;
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
  shader_device_address_ = other.shader_device_address_;
  other.ctx_ = nullptr;
  other.buffer_ = VK_NULL_HANDLE;
  other.allocation_ = VK_NULL_HANDLE;
  other.mapped_ = nullptr;
  other.size_ = 0;
  other.shader_device_address_ = false;
  return *this;
}

auto gpu_buffer::mapped() const noexcept -> std::span<std::byte>
{
  if (mapped_ == nullptr) { return {}; }
  return { static_cast<std::byte *>(mapped_), static_cast<std::size_t>(size_) };
}

auto gpu_buffer::device_address() const -> result<VkDeviceAddress>
{
  if (!shader_device_address_) {
    return std::unexpected(
      make_error(errc::invalid_argument, "vkexec::gpu_buffer was not created with shader_device_address"));
  }
  if (ctx_ == nullptr || ctx_->procs().get_buffer_device_address == nullptr) {
    return std::unexpected(make_error(errc::unsupported, "vkGetBufferDeviceAddress is unavailable"));
  }
  VkBufferDeviceAddressInfo info{};
  info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  info.buffer = buffer_;
  return ctx_->procs().get_buffer_device_address(ctx_->device(), &info);
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
  shader_device_address_ = false;
}

}// namespace vkexec
