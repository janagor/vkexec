#ifndef VKEXEC_TENSOR_HPP
#define VKEXEC_TENSOR_HPP

//! \file
//! Staging-backed typed tensor (host mirror + device storage) for compute ergonomics.

#include <vkexec/copy.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/gpu_buffer.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec/sync_wait.hpp>

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

/**
 * Staging-backed typed tensor for compute ergonomics.
 *
 * Owns a host mirror (`std::vector<T>`), a mapped staging `gpu_buffer`, and a
 * device-local storage `gpu_buffer` created with `shader_device_address`. Shaders
 * bind the device buffer via `storage_binding` / `vk_buffer()`. Host edits go
 * through `span()` / `data()`; transfer them with `upload` / `download` (or
 * `sync_to_device` / `sync_to_host` pass-graph steps).
 *
 * Requires `bufferDeviceAddress` on the context (e.g.
 * `feat::configure<feat::buffer_device_address>`). For simple host-visible SSBOs
 * without staging, prefer `buffer<T>`.
 *
 * @see factory::make_tensor, gpu_buffer, storage_binding, upload_to_device, download_to_host
 */
namespace owned {

  template<typename T>
    requires std::is_trivially_copyable_v<T>
  class tensor
  {
  public:
    tensor() = default;
    ~tensor() = default;

    tensor(tensor const &) = delete;
    auto operator=(tensor const &) -> tensor & = delete;

    tensor(tensor &&) noexcept = default;
    auto operator=(tensor &&) noexcept -> tensor & = default;

    //! Number of `T` elements (0 when empty).
    [[nodiscard]] auto size() const noexcept -> std::size_t { return host_.size(); }
    //! Byte size of the storage buffers.
    [[nodiscard]] auto byte_size() const noexcept -> VkDeviceSize { return size() * sizeof(T); }
    //! Device-local Vulkan buffer handle (null when empty).
    [[nodiscard]] auto vk_buffer() const noexcept -> VkBuffer { return gpu_ ? gpu_->device.handle() : VK_NULL_HANDLE; }

    //! Host mirror pointer (null when empty).
    [[nodiscard]] auto data() noexcept -> T * { return host_.empty() ? nullptr : host_.data(); }
    //! Const host mirror pointer (null when empty).
    [[nodiscard]] auto data() const noexcept -> T const * { return host_.empty() ? nullptr : host_.data(); }

    //! Host span over the CPU mirror.
    [[nodiscard]] auto span() noexcept -> std::span<T> { return { data(), size() }; }
    //! Const host span over the CPU mirror.
    [[nodiscard]] auto span() const noexcept -> std::span<T const> { return { data(), size() }; }

    //! Host-visible staging buffer used for transfers.
    [[nodiscard]] auto staging() noexcept -> owned::gpu_buffer & { return gpu_->staging; }
    //! Const staging buffer.
    [[nodiscard]] auto staging() const noexcept -> owned::gpu_buffer const & { return gpu_->staging; }

    //! Device-local storage buffer bound by shaders.
    [[nodiscard]] auto device() noexcept -> owned::gpu_buffer & { return gpu_->device; }
    //! Const device-local storage buffer.
    [[nodiscard]] auto device() const noexcept -> owned::gpu_buffer const & { return gpu_->device; }

    /**
     * Builds a classic storage descriptor binding for the device buffer.
     *
     * @param binding Descriptor binding index written into the set.
     */
    [[nodiscard]] auto storage_binding(std::uint32_t binding) const noexcept -> vkexec::storage_binding
    {
      return vkexec::storage_binding{
        .buffer = vk_buffer(),
        .byte_size = byte_size(),
        .binding = binding,
      };
    }

    /**
     * Copies the host mirror through staging into the device buffer (blocking).
     *
     * Prefer pass-graph `sync_to_device` when composing sender graphs.
     */
    [[nodiscard]] auto upload(context &ctx) -> status
    { return upload_to_device(ctx, staging(), device(), std::as_bytes(span())); }

    /**
     * Copies the device buffer through staging into the host mirror (blocking).
     *
     * Prefer pass-graph `sync_to_host` when composing sender graphs.
     */
    [[nodiscard]] auto download(context &ctx) -> status
    { return download_to_host(ctx, staging(), device(), std::as_writable_bytes(span())); }

    //! Allocates staging + device storage and wraps `host` (used by `factory::make_tensor`).
    [[nodiscard]] static auto make_allocated(context &ctx, std::vector<T> host) -> result<tensor>;

  private:
    struct gpu_storage
    {
      owned::gpu_buffer staging;
      owned::gpu_buffer device;
    };

    explicit tensor(std::vector<T> host, std::unique_ptr<gpu_storage> gpu) noexcept
      : host_(std::move(host)), gpu_(std::move(gpu))
    {}

    std::vector<T> host_;
    std::unique_ptr<gpu_storage> gpu_;
  };

  template<typename T>
    requires std::is_trivially_copyable_v<T>
  [[nodiscard]] inline auto tensor<T>::make_allocated(context &ctx, std::vector<T> host) -> result<tensor>
  {
    auto const bytes = host.size() * sizeof(T);
    auto staging_buf = try_sync_wait_value(factory::make_gpu_buffer(ctx, bytes, gpu_buffer_memory::staging));
    if (!staging_buf) { return fail(staging_buf.error()); }
    auto device_buf = try_sync_wait_value(factory::make_gpu_buffer(ctx,
      gpu_buffer_create_info{
        .size = bytes,
        .memory = gpu_buffer_memory::device_local,
        .shader_device_address = true,
      }));
    if (!device_buf) { return fail(device_buf.error()); }

    auto staging_map = staging_buf->mapped();
    if (staging_map.size() < bytes) { return fail(errc::unsupported, "vkexec::tensor staging map is too small"); }
    std::memcpy(staging_map.data(), host.data(), static_cast<std::size_t>(bytes));

    auto gpu = std::make_unique<gpu_storage>(gpu_storage{ std::move(*staging_buf), std::move(*device_buf) });
    return tensor{ std::move(host), std::move(gpu) };
  }

}// namespace owned

namespace factory {

  struct make_tensor_t
  {

    /**
     * Allocates `count` elements filled with `fill` on the host mirror.
     *
     * @param ctx Context whose VMA allocator owns the GPU buffers.
     * @param count Element count (must be > 0).
     * @param fill Initial value for every host element.
     */
    template<typename T>
      requires std::is_trivially_copyable_v<T>
    [[nodiscard]] auto operator()(context &ctx, std::size_t count, T fill) const
    {
      return make_sender([&ctx, count, fill]() -> result<owned::tensor<T>> {
        if (count == 0) { return fail(errc::invalid_argument, "vkexec::tensor count must be > 0"); }
        return owned::tensor<T>::make_allocated(ctx, std::vector<T>(count, fill));
      });
    }

    /**
     * Allocates a tensor and copies `values` into the host mirror.
     *
     * @param ctx Context whose VMA allocator owns the GPU buffers.
     * @param values Source elements (must be non-empty).
     */
    template<typename T>
      requires std::is_trivially_copyable_v<T>
    [[nodiscard]] auto operator()(context &ctx, std::span<T const> values) const
    {
      return make_sender([&ctx, values]() -> result<owned::tensor<T>> {
        if (values.empty()) { return fail(errc::invalid_argument, "vkexec::tensor span must be non-empty"); }
        return owned::tensor<T>::make_allocated(ctx, std::vector<T>(values.begin(), values.end()));
      });
    }

    //! Convenience overload that copies from a `std::vector`.
    template<typename T>
      requires std::is_trivially_copyable_v<T>
    [[nodiscard]] auto operator()(context &ctx, std::vector<T> values) const
    {
      return make_sender([&ctx, values = std::move(values)]() mutable -> result<owned::tensor<T>> {
        if (values.empty()) { return fail(errc::invalid_argument, "vkexec::tensor vector must be non-empty"); }
        return owned::tensor<T>::make_allocated(ctx, std::move(values));
      });
    }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_tensor_t make_tensor{};

}// namespace factory

}// namespace vkexec

#endif// VKEXEC_TENSOR_HPP
