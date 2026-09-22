#ifndef VKEXEC_GPU_BUFFER_HPP
#define VKEXEC_GPU_BUFFER_HPP

//! \file
//! Untyped VMA GPU buffers for hybrid / embedder paths.

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
  gpu_buffer_memory memory{ gpu_buffer_memory::host_visible };
  //! Requires bufferDeviceAddress enabled on the device.
  bool shader_device_address{ false };
};

namespace owned {
  class gpu_buffer;
}// namespace owned

namespace factory {

  struct make_gpu_buffer_t
  {

    /**
     * Creates a buffer described by `info`.
     *
     * @param ctx Context whose VMA allocator owns the allocation.
     * @param info Size, memory preset, and optional device address flag.
     * @return Sender that completes with ownership of the buffer.
     */
    //! Creates a buffer of `size` bytes with the given memory preset.
    [[nodiscard]] auto operator()(context &ctx, gpu_buffer_create_info info) const -> sender<owned::gpu_buffer>;
    [[nodiscard]] auto operator()(context &ctx, VkDeviceSize size, gpu_buffer_memory memory) const
      -> sender<owned::gpu_buffer>;
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_gpu_buffer_t make_gpu_buffer{};

}// namespace factory

/**
 * Untyped VMA buffer for hybrid / embedder paths.
 *
 * Distinct from typed `buffer<T>`, which is host-visible storage with element
 * fill. Move-only; destroys via VMA when owned.
 *
 * @see buffer, factory::make_gpu_buffer, upload_to_device
 */
namespace owned {

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
    [[nodiscard]] auto mapped() const noexcept -> std::span<std::byte>;

    /**
     * Returns the buffer device address when created with `shader_device_address`.
     *
     * @return Failure if device address was not requested or the proc is missing.
     */
    [[nodiscard]] auto device_address() const -> result<VkDeviceAddress>;

  private:
    friend struct factory::make_gpu_buffer_t;

    gpu_buffer(context *ctx,
      VkBuffer buffer,
      VmaAllocation allocation,
      void *mapped,
      VkDeviceSize size,
      gpu_buffer_memory memory,
      bool shader_device_address) noexcept;

    auto destroy() noexcept -> void;

    context *ctx_{ nullptr };
    VkBuffer buffer_{ VK_NULL_HANDLE };
    VmaAllocation allocation_{ VK_NULL_HANDLE };
    void *mapped_{ nullptr };
    VkDeviceSize size_{ 0 };
    gpu_buffer_memory memory_{ gpu_buffer_memory::host_visible };
    bool shader_device_address_{ false };
  };

}// namespace owned

}// namespace vkexec

#endif// VKEXEC_GPU_BUFFER_HPP
