#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_BUFFER_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_BUFFER_HPP

#include <vkexec/context.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace vkexec {

/// Host-visible bindless descriptor heap buffer (`VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT`).
class descriptor_heap_buffer
{
public:
  [[nodiscard]] static auto create(context &ctx, VkDeviceSize size) -> detail::sync_sender_fn<descriptor_heap_buffer>;

  ~descriptor_heap_buffer();

  descriptor_heap_buffer(descriptor_heap_buffer const &) = delete;
  auto operator=(descriptor_heap_buffer const &) -> descriptor_heap_buffer & = delete;

  descriptor_heap_buffer(descriptor_heap_buffer &&other) noexcept;
  auto operator=(descriptor_heap_buffer &&other) noexcept -> descriptor_heap_buffer &;

  [[nodiscard]] auto handle() const noexcept -> VkBuffer { return buffer_; }
  [[nodiscard]] auto size() const noexcept -> VkDeviceSize { return size_; }
  [[nodiscard]] auto mapped() const noexcept -> std::span<std::byte>;
  [[nodiscard]] auto device_address() const -> result<VkDeviceAddress>;

private:
  descriptor_heap_buffer(context *ctx,
    VkBuffer buffer,
    VmaAllocation allocation,
    void *mapped,
    VkDeviceSize size) noexcept;

  auto destroy() noexcept -> void;

  context *ctx_{ nullptr };
  VkBuffer buffer_{ VK_NULL_HANDLE };
  VmaAllocation allocation_{ VK_NULL_HANDLE };
  void *mapped_{ nullptr };
  VkDeviceSize size_{ 0 };
};

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_BUFFER_HPP
