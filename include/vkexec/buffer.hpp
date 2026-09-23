#ifndef VKEXEC_BUFFER_HPP
#define VKEXEC_BUFFER_HPP

//! \file
//! Typed host-visible storage buffers allocated via VMA.

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>

#ifndef VKEXEC_ENABLE_EXCEPTIONS
#define VKEXEC_ENABLE_EXCEPTIONS 1
#endif

#include <vkexec/error_helpers.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/sync_wait.hpp>

#include <stdexec/execution.hpp>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace vkexec {

namespace ex = stdexec;

namespace owned {
  template<typename T> class buffer;
}// namespace owned

/**
 * Sender that allocates a host-visible storage buffer and completes with ownership of it.
 *
 * Completes with `set_stopped` when the stop token is already requested, otherwise
 * `set_value(buffer<T>)` or `set_error`.
 *
 * @see factory::make_buffer
 */
template<typename T> struct buffer_allocate_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(owned::buffer<T>), ex::set_error_t(error), ex::set_stopped_t()>;

  context *ctx{ nullptr };
  std::size_t count{ 0 };
  T fill{};

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  template<class Receiver> struct op_state
  {
    context *ctx{ nullptr };
    std::size_t count{ 0 };
    T fill{};
    Receiver receiver;

    auto start() noexcept -> void
    {
      auto const token = ex::get_stop_token(ex::get_env(receiver));
      if constexpr (!ex::unstoppable_token<std::remove_cvref_t<decltype(token)>>) {
        if (token.stop_requested()) {
          ex::set_stopped(std::move(receiver));
          return;
        }
      }

      std::optional<result<owned::buffer<T>>> allocated;
#if VKEXEC_ENABLE_EXCEPTIONS
      try {
        allocated.emplace(owned::buffer<T>::make_allocated(*ctx, count, fill));
      } catch (...) {
        ex::set_error(std::move(receiver), unexpected_exception_error());
        return;
      }
#else
      allocated.emplace(owned::buffer<T>::make_allocated(*ctx, count, fill));
#endif

      auto &allocation_result = *allocated;
      if (allocation_result) {
        ex::set_value(std::move(receiver), expected_take(allocation_result));
      } else {
        ex::set_error(std::move(receiver), std::move(allocation_result.error()));
      }
    }
  };

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) & -> op_state<Receiver>
  {
    return op_state<Receiver>{
      ctx,
      count,
      fill,
      std::move(receiver),
    };
  }

  template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) && -> op_state<Receiver>
  {
    return op_state<Receiver>{
      ctx,
      count,
      std::move(fill),
      std::move(receiver),
    };
  }
};

/**
 * Typed host-visible `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` buffer.
 *
 * Elements are trivially copyable `T`. Creation fills every element with the
 * provided value. Persistently mapped for host access via `data()`.
 *
 * Prefer `factory::make_buffer` over constructing directly.
 *
 * @see factory::make_buffer, buffer_allocate_sender, gpu_buffer
 */
namespace owned {

  template<typename T> class buffer
  {
  public:
    ~buffer()
    {
      if (ctx_ != nullptr && ctx_->allocator() != VK_NULL_HANDLE && buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(ctx_->allocator(), buffer_, allocation_);
      }
    }

    buffer(buffer const &) = delete;
    auto operator=(buffer const &) -> buffer & = delete;

    buffer(buffer &&other) noexcept
      : ctx_(other.ctx_), buffer_(other.buffer_), allocation_(other.allocation_), mapped_(other.mapped_),
        count_(other.count_), name_(std::move(other.name_))
    {
      other.ctx_ = nullptr;
      other.buffer_ = VK_NULL_HANDLE;
      other.allocation_ = VK_NULL_HANDLE;
      other.mapped_ = nullptr;
    }

    auto operator=(buffer &&other) noexcept -> buffer &
    {
      if (this == &other) { return *this; }
      if (ctx_ != nullptr && ctx_->allocator() != VK_NULL_HANDLE && buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(ctx_->allocator(), buffer_, allocation_);
      }
      ctx_ = other.ctx_;
      buffer_ = other.buffer_;
      allocation_ = other.allocation_;
      mapped_ = other.mapped_;
      count_ = other.count_;
      name_ = std::move(other.name_);
      other.ctx_ = nullptr;
      other.buffer_ = VK_NULL_HANDLE;
      other.allocation_ = VK_NULL_HANDLE;
      other.mapped_ = nullptr;
      other.count_ = 0;
      return *this;
    }

    //! Persistently mapped host pointer to `size()` elements.
    [[nodiscard]] auto data() noexcept -> T * { return static_cast<T *>(mapped_); }
    //! Persistently mapped const host pointer.
    [[nodiscard]] auto data() const noexcept -> T const * { return static_cast<T const *>(mapped_); }
    //! Number of `T` elements.
    [[nodiscard]] auto size() const noexcept -> std::size_t { return count_; }
    //! Byte size of the allocation.
    [[nodiscard]] auto byte_size() const noexcept -> VkDeviceSize { return count_ * sizeof(T); }
    //! Vulkan buffer handle.
    [[nodiscard]] auto vk_buffer() const noexcept -> VkBuffer { return buffer_; }
    //! Debug name assigned at allocation.
    [[nodiscard]] auto name() const noexcept -> std::string const & { return name_; }

  private:
    friend struct buffer_allocate_sender<T>;

    struct owned_tag
    {
    };

    buffer(owned_tag /*tag*/,
      context *ctx,
      VkBuffer handle,
      VmaAllocation allocation,
      void *mapped,
      std::size_t count,
      std::string name) noexcept
      : ctx_(ctx), buffer_(handle), allocation_(allocation), mapped_(mapped), count_(count), name_(std::move(name))
    {}

    [[nodiscard]] static auto make_allocated(context &ctx, std::size_t count, T fill) -> result<buffer>
    {
      static_assert(std::is_trivially_copyable_v<T>);
      static_assert(std::is_nothrow_copy_assignable_v<T>);
      if (count == 0) { return fail(errc::invalid_argument, "vkexec::buffer count must be > 0"); }

      std::string buffer_name = "buf" + std::to_string(next_name_id());

      auto const bytes = count * sizeof(T);

      VkBufferCreateInfo bci{};
      bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bci.size = bytes;
      bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

      VmaAllocationCreateInfo aci{};
      aci.usage = VMA_MEMORY_USAGE_AUTO;
      aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
      aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

      VkBuffer handle{ VK_NULL_HANDLE };
      VmaAllocation allocation{ VK_NULL_HANDLE };
      VmaAllocationInfo ainfo{};
      if (VkResult const created = vmaCreateBuffer(ctx.allocator(), &bci, &aci, &handle, &allocation, &ainfo);
        created != VK_SUCCESS) {
        return fail(created, "vmaCreateBuffer failed");
      }
      if (ainfo.pMappedData == nullptr) {
        vmaDestroyBuffer(ctx.allocator(), handle, allocation);
        return fail(errc::io_error, "vmaCreateBuffer did not map host-visible memory");
      }

      auto *const elems = static_cast<T *>(ainfo.pMappedData);
      for (T &elem : std::span<T>{ elems, count }) { elem = fill; }

      return buffer(owned_tag{}, &ctx, handle, allocation, ainfo.pMappedData, count, std::move(buffer_name));
    }

    static auto next_name_id() -> int
    {
      static int name_id = 0;
      return name_id++;
    }

    context *ctx_{ nullptr };
    VkBuffer buffer_{ VK_NULL_HANDLE };
    VmaAllocation allocation_{ VK_NULL_HANDLE };
    void *mapped_{ nullptr };
    std::size_t count_{ 0 };
    std::string name_;
  };

}// namespace owned

namespace factory {

  /**
   * Returns a sender that allocates `count` elements filled with `fill`.
   *
   * @param ctx Context whose VMA allocator owns the buffer.
   * @param count Element count (must be > 0).
   * @param fill Initial value written to every element.
   */
  struct make_buffer_t
  {
    template<typename T>
    [[nodiscard]] auto operator()(context &ctx, std::size_t count, T fill) const -> buffer_allocate_sender<T>
    { return buffer_allocate_sender<T>{ .ctx = &ctx, .count = count, .fill = std::move(fill) }; }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_buffer_t make_buffer{};

}// namespace factory

}// namespace vkexec

#endif// VKEXEC_BUFFER_HPP
