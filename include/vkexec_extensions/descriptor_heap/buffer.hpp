#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_BUFFER_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_BUFFER_HPP

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace vkexec {

class descriptor_heap_buffer;

namespace factory {

  /**
   * Creates a host-visible, host-coherent descriptor heap buffer of `size` bytes.
   *
   * @param ctx Context whose VMA allocator owns the allocation.
   * @param size Byte size (must accommodate descriptors + reserved range).
   */
  [[nodiscard]] auto descriptor_heap_buffer(::vkexec::context &ctx, VkDeviceSize size)
    -> sender<::vkexec::descriptor_heap_buffer>;

}// namespace factory

/**
 * Host-visible, host-coherent bindless descriptor heap buffer
 * (`VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT`).
 *
 * Persistently mapped for host descriptor writes; coherent memory is required so
 * continuous host writes are GPU-visible without explicit flushes. Use
 * `device_address()` when binding the heap on the GPU. `flush()` is a no-op when
 * the allocation is coherent (belt-and-suspenders for non-coherent fallbacks).
 *
 * @see query_descriptor_heap_layout, cmd_bind_resource_heap, factory::descriptor_heap_buffer
 */
class descriptor_heap_buffer
{
public:
  ~descriptor_heap_buffer();

  descriptor_heap_buffer(descriptor_heap_buffer const &) = delete;
  auto operator=(descriptor_heap_buffer const &) -> descriptor_heap_buffer & = delete;

  descriptor_heap_buffer(descriptor_heap_buffer &&other) noexcept;
  auto operator=(descriptor_heap_buffer &&other) noexcept -> descriptor_heap_buffer &;

  //! Vulkan buffer handle (null after move).
  [[nodiscard]] auto handle() const noexcept -> VkBuffer { return buffer_; }
  //! Allocated size in bytes.
  [[nodiscard]] auto size() const noexcept -> VkDeviceSize { return size_; }
  //! Persistently mapped host bytes for descriptor writes.
  [[nodiscard]] auto mapped() const noexcept -> std::span<std::byte>;
  //! Device address for `cmd_bind_resource_heap` (requires buffer device address).
  [[nodiscard]] auto device_address() const -> result<VkDeviceAddress>;

  /**
   * Flushes mapped host writes when the allocation is not host-coherent.
   *
   * No-op for coherent allocations (the create default). Safe to call after
   * descriptor writes if an embedder wants an explicit visibility barrier.
   */
  [[nodiscard]] auto flush() const -> status;

private:
  friend sender<::vkexec::descriptor_heap_buffer> factory::descriptor_heap_buffer(::vkexec::context &ctx,
    VkDeviceSize size);

  descriptor_heap_buffer(context *ctx,
    VkBuffer buffer,
    VmaAllocation allocation,
    void *mapped,
    VkDeviceSize size,
    bool host_coherent) noexcept;

  auto destroy() noexcept -> void;

  context *ctx_{ nullptr };
  VkBuffer buffer_{ VK_NULL_HANDLE };
  VmaAllocation allocation_{ VK_NULL_HANDLE };
  void *mapped_{ nullptr };
  VkDeviceSize size_{ 0 };
  bool host_coherent_{ true };
};

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_BUFFER_HPP
