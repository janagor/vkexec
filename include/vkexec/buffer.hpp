#ifndef VKEXEC_BUFFER_HPP
#define VKEXEC_BUFFER_HPP

//! \file
//! Typed host-visible storage built on a buffer allocator.

#include <vkexec/detail/normalize_errors.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>

#include <stdexec/execution.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

namespace vkexec {

template<class A>
concept mapped_buffer_allocator = buffer_allocator<A> && flushable_buffer_resource<typename A::buffer_type>;

template<typename T, flushable_buffer_resource B> class buffer
{
public:
  using resource_type = B;

  buffer(buffer const &) = delete;
  auto operator=(buffer const &) -> buffer & = delete;
  buffer(buffer &&) noexcept(std::is_nothrow_move_constructible_v<resource_type>) = default;
  auto operator=(buffer &&) noexcept(std::is_nothrow_move_assignable_v<resource_type>) -> buffer & = default;
  ~buffer() = default;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  [[nodiscard]] auto data() noexcept -> T * { return reinterpret_cast<T *>(resource_.mapped().data()); }
  [[nodiscard]] auto data() const noexcept -> T const *
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  { return reinterpret_cast<T const *>(resource_.mapped().data()); }
  [[nodiscard]] auto size() const noexcept -> std::size_t { return count_; }
  [[nodiscard]] auto byte_size() const noexcept -> VkDeviceSize { return count_ * sizeof(T); }
  [[nodiscard]] auto handle() const noexcept -> VkBuffer { return resource_.handle(); }
  [[nodiscard]] auto vk_buffer() const noexcept -> VkBuffer { return handle(); }
  [[nodiscard]] auto resource() noexcept -> resource_type & { return resource_; }
  [[nodiscard]] auto resource() const noexcept -> resource_type const & { return resource_; }

  [[nodiscard]] static auto make_initialized(resource_type resource, std::size_t count, T fill) -> result<buffer>
  {
    static_assert(std::is_trivially_copyable_v<T>);
    auto const bytes = count * sizeof(T);
    auto mapped = resource.mapped();
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
    if (resource.size() < bytes || mapped.size() < bytes
        || reinterpret_cast<std::uintptr_t>(mapped.data()) % alignof(T) != 0) {
      return fail(errc::unsupported, "allocator did not provide aligned mapped storage");
    }
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
    for (std::size_t index = 0; index < count; ++index) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      std::memcpy(mapped.data() + (index * sizeof(T)), &fill, sizeof(T));
    }
    if (auto flushed = resource.flush(); !flushed) { return fail(flushed.error()); }
    return buffer{ std::move(resource), count };
  }

private:
  explicit buffer(resource_type resource, std::size_t count) noexcept(
    std::is_nothrow_move_constructible_v<resource_type>)
    : resource_(std::move(resource)), count_(count)
  {}

  resource_type resource_;
  std::size_t count_{};
};

namespace factory {
  struct make_buffer_t
  {
    template<typename T, mapped_buffer_allocator A>
    [[nodiscard]] auto operator()(A &allocator, std::size_t count, T fill) const
    {
      using resource_type = A::buffer_type;
      auto request = make_sender([count]() -> result<VkDeviceSize> {
        if (count == 0 || count > std::numeric_limits<VkDeviceSize>::max() / sizeof(T)) {
          return fail(errc::invalid_argument, "buffer element count is invalid");
        }
        return count * sizeof(T);
      });
      auto composed =
        std::move(request)
        | stdexec::let_value([&allocator, count, fill = std::move(fill)](VkDeviceSize &bytes) mutable -> auto {
            return allocate_buffer(allocator,
                     buffer_create_info{
                       .size = bytes,
                       .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       .memory = memory_domain::host_visible,
                       .alignment = alignof(T),
                     })
                   | stdexec::let_value([count, fill = std::move(fill)](resource_type &resource) mutable -> auto {
                       return make_sender(
                         [&resource, count, fill = std::move(fill)]() mutable -> result<buffer<T, resource_type>> {
                           return buffer<T, resource_type>::make_initialized(
                             std::move(resource), count, std::move(fill));
                         });
                     });
          });
      return ::vkexec::detail::normalize_errors(std::move(composed));
    }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_buffer_t make_buffer{};
}// namespace factory

}// namespace vkexec

#endif// VKEXEC_BUFFER_HPP
