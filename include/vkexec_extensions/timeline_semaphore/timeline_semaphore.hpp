#ifndef VKEXEC_EXTENSIONS_TIMELINE_SEMAPHORE_TIMELINE_SEMAPHORE_HPP
#define VKEXEC_EXTENSIONS_TIMELINE_SEMAPHORE_TIMELINE_SEMAPHORE_HPP

#include <vkexec/context.hpp>
#include <vkexec/result.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>

namespace vkexec {

class timeline_semaphore;

namespace detail {

  [[nodiscard]] auto make_timeline_semaphore(context &ctx, std::uint64_t initial_value = 0) -> result<timeline_semaphore>;

}// namespace detail

/// RAII timeline semaphore (`VK_SEMAPHORE_TYPE_TIMELINE`).
class timeline_semaphore
{
public:
  [[nodiscard]] static auto create(context &ctx, std::uint64_t initial_value = 0)
    -> detail::sync_sender_fn<timeline_semaphore>;

  ~timeline_semaphore();

  timeline_semaphore(timeline_semaphore const &) = delete;
  auto operator=(timeline_semaphore const &) -> timeline_semaphore & = delete;

  timeline_semaphore(timeline_semaphore &&other) noexcept;
  auto operator=(timeline_semaphore &&other) noexcept -> timeline_semaphore &;

  [[nodiscard]] auto handle() const noexcept -> VkSemaphore { return semaphore_; }

  /// Host wait until the semaphore reaches at least `value` (no-op when value == 0).
  [[nodiscard]] auto wait(std::uint64_t value) const -> status;

private:
  friend auto detail::make_timeline_semaphore(context &ctx, std::uint64_t initial_value) -> result<timeline_semaphore>;

  timeline_semaphore(context *ctx, VkSemaphore semaphore) noexcept;
  auto destroy() noexcept -> void;

  context *ctx_{ nullptr };
  VkSemaphore semaphore_{ VK_NULL_HANDLE };
};

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_TIMELINE_SEMAPHORE_TIMELINE_SEMAPHORE_HPP
