#ifndef VKEXEC_TENSOR_HPP
#define VKEXEC_TENSOR_HPP

//! \file
//! Typed tensor helper over a host-visible storage buffer (Kompute-inspired).

#include <vkexec/buffer.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sync_wait.hpp>

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

/**
 * Typed host-visible storage tensor for compute ergonomics.
 *
 * Owns a `buffer<T>` (persistently mapped). Prefer this when you want a
 * Kompute-like typed blob with `storage_binding` helpers. Device-local staging
 * sync lives in later APIs (`sync_to_device` / `sync_to_host`).
 *
 * @see buffer, storage_binding
 */
template<typename T>
  requires std::is_trivially_copyable_v<T>
class tensor
{
public:
  /**
   * Allocates `count` elements filled with `fill`.
   *
   * @param ctx Context whose VMA allocator owns the buffer.
   * @param count Element count (must be > 0).
   * @param fill Initial value for every element.
   */
  [[nodiscard]] static auto create(context &ctx, std::size_t count, T fill = T{}) -> detail::sync_sender_fn<tensor>
  {
    return detail::make_sync_sender_fn<tensor>([&ctx, count, fill]() -> result<tensor> {
      auto allocated = try_sync_wait_value(buffer<T>::allocate(ctx, count, fill));
      if (!allocated) { return fail(allocated.error()); }
      return tensor{ std::move(*allocated) };
    });
  }

  /**
   * Allocates a tensor and copies `values` into it.
   *
   * @param ctx Context whose VMA allocator owns the buffer.
   * @param values Source elements (must be non-empty).
   */
  [[nodiscard]] static auto create(context &ctx, std::span<T const> values) -> detail::sync_sender_fn<tensor>
  {
    return detail::make_sync_sender_fn<tensor>([&ctx, values]() -> result<tensor> {
      if (values.empty()) { return fail(errc::invalid_argument, "vkexec::tensor span must be non-empty"); }
      auto allocated = try_sync_wait_value(buffer<T>::allocate(ctx, values.size(), T{}));
      if (!allocated) { return fail(allocated.error()); }
      tensor owned{ std::move(*allocated) };
      std::memcpy(owned.data(), values.data(), values.size_bytes());
      return owned;
    });
  }

  //! Convenience overload that copies from a `std::vector`.
  [[nodiscard]] static auto create(context &ctx, std::vector<T> values) -> detail::sync_sender_fn<tensor>
  {
    return detail::make_sync_sender_fn<tensor>([&ctx, values = std::move(values)]() -> result<tensor> {
      if (values.empty()) { return fail(errc::invalid_argument, "vkexec::tensor vector must be non-empty"); }
      auto allocated = try_sync_wait_value(buffer<T>::allocate(ctx, values.size(), T{}));
      if (!allocated) { return fail(allocated.error()); }
      tensor owned{ std::move(*allocated) };
      std::memcpy(owned.data(), values.data(), values.size() * sizeof(T));
      return owned;
    });
  }

  tensor() = default;
  ~tensor() = default;

  tensor(tensor const &) = delete;
  auto operator=(tensor const &) -> tensor & = delete;

  tensor(tensor &&) noexcept = default;
  auto operator=(tensor &&) noexcept -> tensor & = default;

  //! Number of `T` elements (0 when empty).
  [[nodiscard]] auto size() const noexcept -> std::size_t { return storage_ ? storage_->size() : 0; }
  //! Byte size of the storage buffer.
  [[nodiscard]] auto byte_size() const noexcept -> VkDeviceSize
  { return static_cast<VkDeviceSize>(size() * sizeof(T)); }
  //! Vulkan buffer handle (null when empty).
  [[nodiscard]] auto vk_buffer() const noexcept -> VkBuffer
  { return storage_ ? storage_->vk_buffer() : VK_NULL_HANDLE; }

  //! Persistently mapped host pointer (null when empty).
  [[nodiscard]] auto data() noexcept -> T * { return storage_ ? storage_->data() : nullptr; }
  //! Persistently mapped const host pointer (null when empty).
  [[nodiscard]] auto data() const noexcept -> T const * { return storage_ ? storage_->data() : nullptr; }

  //! Host span over mapped elements.
  [[nodiscard]] auto span() noexcept -> std::span<T> { return { data(), size() }; }
  //! Const host span over mapped elements.
  [[nodiscard]] auto span() const noexcept -> std::span<T const> { return { data(), size() }; }

  //! Owned host-visible storage buffer.
  [[nodiscard]] auto storage() noexcept -> buffer<T> & { return *storage_; }
  //! Const owned storage buffer.
  [[nodiscard]] auto storage() const noexcept -> buffer<T> const & { return *storage_; }

  /**
   * Builds a classic storage descriptor binding for this tensor.
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

private:
  explicit tensor(buffer<T> storage) noexcept : storage_(std::move(storage)) {}

  std::optional<buffer<T>> storage_;
};

}// namespace vkexec

#endif// VKEXEC_TENSOR_HPP
