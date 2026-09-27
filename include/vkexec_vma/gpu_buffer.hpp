#ifndef VKEXEC_VMA_GPU_BUFFER_HPP
#define VKEXEC_VMA_GPU_BUFFER_HPP

//! \file
//! Untyped VMA GPU buffers for hybrid / embedder paths.

#include <vkexec/error.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec_vma/allocator.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace vkexec {

namespace vma {

  /**
   * Memory / usage preset for untyped GPU buffers.
   *
   * @see gpu_buffer, gpu_buffer_create_info
   */
  enum class gpu_buffer_memory : std::uint8_t {
    //! Host-visible storage; persistently mapped for sequential host writes.
    host_visible,
    //! Device-local storage with transfer + indirect usage.
    device_local,
    //! Host-visible transfer destination for GPU->CPU readback.
    staging,
  };

  /**
   * Creation parameters for `factory::make_gpu_buffer`.
   *
   * @param size Byte size of the buffer (must be > 0).
   * @param memory Memory/usage preset.
   * @param shader_device_address When true, requires `bufferDeviceAddress` on the device.
   */
  struct gpu_buffer_create_info
  {
    VkDeviceSize size{ 0 };
    //! Extra Vulkan usage bits, e.g. vertex or index-buffer use.
    VkBufferUsageFlags usage{ 0 };
    gpu_buffer_memory memory{ gpu_buffer_memory::host_visible };
    //! Requires bufferDeviceAddress enabled on the device.
    bool shader_device_address{ false };
    VkDeviceSize alignment{};
  };

}// namespace vma

namespace vma {
  class gpu_buffer;
}// namespace vma

namespace detail {

  struct make_gpu_buffer_factory
  {
    vma::allocator *allocator;
    vma::gpu_buffer_create_info info;

    [[nodiscard]] auto operator()() const -> result<vma::gpu_buffer>;
  };

}// namespace detail

/**
 * Untyped VMA buffer for hybrid / embedder paths.
 *
 * Distinct from typed `buffer<T>`, which is host-visible storage with element
 * fill. Move-only; destroys via VMA when owned.
 *
 * @see buffer, factory::make_gpu_buffer, upload_to_device
 */
namespace vma {

  class gpu_buffer
  {
  public:
    ~gpu_buffer();

    gpu_buffer(gpu_buffer const &) = delete;
    auto operator=(gpu_buffer const &) -> gpu_buffer & = delete;

    gpu_buffer(gpu_buffer &&other) noexcept;
    auto operator=(gpu_buffer &&other) noexcept -> gpu_buffer &;

    //! Vulkan buffer handle (null after move).
    [[nodiscard]] auto handle() const noexcept -> VkBuffer { return buffer_; }
    //! Allocated size in bytes.
    [[nodiscard]] auto size() const noexcept -> VkDeviceSize { return size_; }
    //! Memory preset used at creation.
    [[nodiscard]] auto memory() const noexcept -> gpu_buffer_memory { return memory_; }

    /**
     * Returns the persistently mapped host span when the buffer is host-visible.
     *
     * Empty when the buffer is device-local or unmapped.
     */
    [[nodiscard]] auto mapped() noexcept -> std::span<std::byte>;
    [[nodiscard]] auto mapped() const noexcept -> std::span<std::byte const>;
    [[nodiscard]] auto flush() const -> status;
    [[nodiscard]] auto invalidate() const -> status;

    /**
     * Returns the buffer device address when created with `shader_device_address`.
     *
     * @return Failure if device address was not requested or the proc is missing.
     */
    [[nodiscard]] auto device_address() const -> result<VkDeviceAddress>;

  private:
    friend struct ::vkexec::detail::make_gpu_buffer_factory;

    gpu_buffer(context *ctx,
      VmaAllocator allocator,
      VkBuffer buffer,
      VmaAllocation allocation,
      void *mapped,
      VkDeviceSize size,
      gpu_buffer_memory memory,
      bool shader_device_address) noexcept;

    auto destroy() noexcept -> void;

    context *ctx_{ nullptr };
    VmaAllocator allocator_{ VK_NULL_HANDLE };
    VkBuffer buffer_{ VK_NULL_HANDLE };
    VmaAllocation allocation_{ VK_NULL_HANDLE };
    void *mapped_{ nullptr };
    VkDeviceSize size_{ 0 };
    gpu_buffer_memory memory_{ gpu_buffer_memory::host_visible };
    bool shader_device_address_{ false };
  };

}// namespace vma

}// namespace vkexec

namespace vkexec::vma {

using buffer_memory = gpu_buffer_memory;

namespace factory {
  struct make_gpu_buffer_t
  {
    [[nodiscard]] auto operator()(allocator &allocator, gpu_buffer_create_info info) const
    { return make_sender(::vkexec::detail::make_gpu_buffer_factory{ .allocator = &allocator, .info = info }); }

    [[nodiscard]] auto operator()(allocator &allocator, VkDeviceSize size, buffer_memory memory) const
    { return (*this)(allocator, gpu_buffer_create_info{ .size = size, .memory = memory }); }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_gpu_buffer_t make_gpu_buffer{};
}// namespace factory

[[nodiscard]] inline auto tag_invoke(allocate_buffer_t /*tag*/, allocator &alloc, buffer_create_info info)
{
  gpu_buffer_memory memory{};
  switch (info.memory) {
  case memory_domain::host_visible:
    memory = gpu_buffer_memory::host_visible;
    break;
  case memory_domain::device_local:
    memory = gpu_buffer_memory::device_local;
    break;
  case memory_domain::staging:
    memory = gpu_buffer_memory::staging;
    break;
  }
  return factory::make_gpu_buffer(alloc,
    gpu_buffer_create_info{
      .size = info.size,
      .usage = info.usage,
      .memory = memory,
      .shader_device_address = info.shader_device_address,
      .alignment = info.alignment,
    });
}

static_assert(buffer_allocator<allocator>);

}// namespace vkexec::vma

#endif// VKEXEC_VMA_GPU_BUFFER_HPP
