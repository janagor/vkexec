#ifndef VKEXEC_TENSOR_HPP
#define VKEXEC_TENSOR_HPP

//! \file
//! Staging-backed typed tensor (host mirror + device storage) for compute ergonomics.

#include <vkexec/copy.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/gpu_buffer.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
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
 * device-local storage `gpu_buffer`. Shaders bind the device buffer via
 * `storage_binding` / `vk_buffer()`. Host edits go through `span()` / `data()`;
 * transfer them with `upload` / `download` (or later `sync_to_device` /
 * `sync_to_host` pass-graph steps).
 *
 * For simple host-visible SSBOs without staging, prefer `buffer<T>`.
 *
 * @see gpu_buffer, storage_binding, upload_to_device, download_to_host
 */
template<typename T>
  requires std::is_trivially_copyable_v<T>
class tensor
{
public:
  /**
   * Allocates `count` elements filled with `fill` on the host mirror.
   *
   * @param ctx Context whose VMA allocator owns the GPU buffers.
   * @param count Element count (must be > 0).
   * @param fill Initial value for every host element.
   */
  [[nodiscard]] static auto create(context &ctx, std::size_t count, T fill = T{}) -> detail::sync_sender_fn<tensor>
  {
    return detail::make_sync_sender_fn<tensor>([&ctx, count, fill]() -> result<tensor> {
      if (count == 0) { return fail(errc::invalid_argument, "vkexec::tensor count must be > 0"); }
      return make_allocated(ctx, std::vector<T>(count, fill));
    });
  }

  /**
   * Allocates a tensor and copies `values` into the host mirror.
   *
   * @param ctx Context whose VMA allocator owns the GPU buffers.
   * @param values Source elements (must be non-empty).
   */
  [[nodiscard]] static auto create(context &ctx, std::span<T const> values) -> detail::sync_sender_fn<tensor>
  {
    return detail::make_sync_sender_fn<tensor>([&ctx, values]() -> result<tensor> {
      if (values.empty()) { return fail(errc::invalid_argument, "vkexec::tensor span must be non-empty"); }
      return make_allocated(ctx, std::vector<T>(values.begin(), values.end()));
    });
  }

  //! Convenience overload that copies from a `std::vector`.
  [[nodiscard]] static auto create(context &ctx, std::vector<T> values) -> detail::sync_sender_fn<tensor>
  {
    return detail::make_sync_sender_fn<tensor>([&ctx, values = std::move(values)]() mutable -> result<tensor> {
      if (values.empty()) { return fail(errc::invalid_argument, "vkexec::tensor vector must be non-empty"); }
      return make_allocated(ctx, std::move(values));
    });
  }

  tensor() = default;
  ~tensor() = default;

  tensor(tensor const &) = delete;
  auto operator=(tensor const &) -> tensor & = delete;

  tensor(tensor &&) noexcept = default;
  auto operator=(tensor &&) noexcept -> tensor & = default;

  //! Number of `T` elements (0 when empty).
  [[nodiscard]] auto size() const noexcept -> std::size_t { return host_.size(); }
  //! Byte size of the storage buffers.
  [[nodiscard]] auto byte_size() const noexcept -> VkDeviceSize
  { return static_cast<VkDeviceSize>(size() * sizeof(T)); }
  //! Device-local Vulkan buffer handle (null when empty).
  [[nodiscard]] auto vk_buffer() const noexcept -> VkBuffer
  { return gpu_ ? gpu_->device.handle() : VK_NULL_HANDLE; }

  //! Host mirror pointer (null when empty).
  [[nodiscard]] auto data() noexcept -> T * { return host_.empty() ? nullptr : host_.data(); }
  //! Const host mirror pointer (null when empty).
  [[nodiscard]] auto data() const noexcept -> T const * { return host_.empty() ? nullptr : host_.data(); }

  //! Host span over the CPU mirror.
  [[nodiscard]] auto span() noexcept -> std::span<T> { return { data(), size() }; }
  //! Const host span over the CPU mirror.
  [[nodiscard]] auto span() const noexcept -> std::span<T const> { return { data(), size() }; }

  //! Host-visible staging buffer used for transfers.
  [[nodiscard]] auto staging() noexcept -> gpu_buffer & { return gpu_->staging; }
  //! Const staging buffer.
  [[nodiscard]] auto staging() const noexcept -> gpu_buffer const & { return gpu_->staging; }

  //! Device-local storage buffer bound by shaders.
  [[nodiscard]] auto device() noexcept -> gpu_buffer & { return gpu_->device; }
  //! Const device-local storage buffer.
  [[nodiscard]] auto device() const noexcept -> gpu_buffer const & { return gpu_->device; }

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

private:
  struct gpu_storage
  {
    gpu_buffer staging;
    gpu_buffer device;
  };

  [[nodiscard]] static auto make_allocated(context &ctx, std::vector<T> host) -> result<tensor>
  {
    auto const bytes = static_cast<VkDeviceSize>(host.size() * sizeof(T));
    auto staging_buf = try_sync_wait_value(gpu_buffer::create(ctx, bytes, gpu_buffer_memory::staging));
    if (!staging_buf) { return fail(staging_buf.error()); }
    auto device_buf = try_sync_wait_value(gpu_buffer::create(ctx, bytes, gpu_buffer_memory::device_local));
    if (!device_buf) { return fail(device_buf.error()); }

    auto staging_map = staging_buf->mapped();
    if (staging_map.size() < bytes) {
      return fail(errc::unsupported, "vkexec::tensor staging map is too small");
    }
    std::memcpy(staging_map.data(), host.data(), static_cast<std::size_t>(bytes));

    auto gpu = std::make_unique<gpu_storage>(gpu_storage{ std::move(*staging_buf), std::move(*device_buf) });
    return tensor{ std::move(host), std::move(gpu) };
  }

  explicit tensor(std::vector<T> host, std::unique_ptr<gpu_storage> gpu) noexcept
    : host_(std::move(host)), gpu_(std::move(gpu))
  {}

  std::vector<T> host_;
  std::unique_ptr<gpu_storage> gpu_;
};

}// namespace vkexec

#endif// VKEXEC_TENSOR_HPP
