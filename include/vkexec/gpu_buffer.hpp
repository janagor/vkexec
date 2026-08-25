#ifndef VKEXEC_GPU_BUFFER_HPP
#define VKEXEC_GPU_BUFFER_HPP

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace vkexec {

/// Memory / usage preset for untyped GPU buffers (non-eDSL path).
enum class gpu_buffer_memory : std::uint8_t {
  /// Host-visible storage; persistently mapped for sequential host writes.
  host_visible,
  /// Device-local storage with transfer + indirect usage.
  device_local,
  /// Host-visible transfer destination for GPU→CPU readback.
  staging,
  /// Host-visible descriptor-heap buffer (`VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT`).
  descriptor_heap,
};

/// Untyped VMA buffer for hybrid / embedders (e.g. vkgsplat). Distinct from eDSL `buffer<T>`.
class gpu_buffer
{
public:
  [[nodiscard]] static auto create(context &ctx, VkDeviceSize size, gpu_buffer_memory memory) -> gpu_buffer;

  ~gpu_buffer();

  gpu_buffer(gpu_buffer const &) = delete;
  auto operator=(gpu_buffer const &) -> gpu_buffer & = delete;

  gpu_buffer(gpu_buffer &&other) noexcept;
  auto operator=(gpu_buffer &&other) noexcept -> gpu_buffer &;

  [[nodiscard]] auto handle() const noexcept -> VkBuffer { return buffer_; }
  [[nodiscard]] auto size() const noexcept -> VkDeviceSize { return size_; }
  [[nodiscard]] auto memory() const noexcept -> gpu_buffer_memory { return memory_; }
  [[nodiscard]] auto mapped() const noexcept -> std::span<std::byte>;

private:
  gpu_buffer(context *ctx,
    VkBuffer buffer,
    VmaAllocation allocation,
    void *mapped,
    VkDeviceSize size,
    gpu_buffer_memory memory) noexcept;

  auto destroy() noexcept -> void;

  context *ctx_{ nullptr };
  VkBuffer buffer_{ VK_NULL_HANDLE };
  VmaAllocation allocation_{ VK_NULL_HANDLE };
  void *mapped_{ nullptr };
  VkDeviceSize size_{ 0 };
  gpu_buffer_memory memory_{ gpu_buffer_memory::host_visible };
};

}// namespace vkexec

#endif// VKEXEC_GPU_BUFFER_HPP
