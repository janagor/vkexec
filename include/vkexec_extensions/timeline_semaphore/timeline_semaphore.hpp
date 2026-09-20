#ifndef VKEXEC_EXTENSIONS_TIMELINE_SEMAPHORE_TIMELINE_SEMAPHORE_HPP
#define VKEXEC_EXTENSIONS_TIMELINE_SEMAPHORE_TIMELINE_SEMAPHORE_HPP

//! \file
//! RAII timeline semaphore (`VK_SEMAPHORE_TYPE_TIMELINE`).

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>

namespace vkexec {

class timeline_semaphore;

namespace detail {

  //! Creates a timeline semaphore with `initial_value` on `ctx`'s device.
  [[nodiscard]] auto make_timeline_semaphore(context &ctx, std::uint64_t initial_value = 0)
    -> result<timeline_semaphore>;

}// namespace detail

/**
 * RAII timeline semaphore (`VK_SEMAPHORE_TYPE_TIMELINE`).
 *
 * Requires timeline semaphore support on the device (see `feat::timeline_semaphore`).
 * Destroyed on the context device when this object is destroyed or moved-from.
 *
 * @see frame_ring, feat::timeline_semaphore
 */
class timeline_semaphore
{
public:
  /**
   * Creates a timeline semaphore with the given initial counter value.
   *
   * @param ctx Context that owns the device.
   * @param initial_value Starting timeline value (often 0).
   */
  [[nodiscard]] static auto create(context &ctx, std::uint64_t initial_value = 0) -> sender<timeline_semaphore>;

  ~timeline_semaphore();

  timeline_semaphore(timeline_semaphore const &) = delete;
  auto operator=(timeline_semaphore const &) -> timeline_semaphore & = delete;

  timeline_semaphore(timeline_semaphore &&other) noexcept;
  auto operator=(timeline_semaphore &&other) noexcept -> timeline_semaphore &;

  //! Vulkan semaphore handle (null after move).
  [[nodiscard]] auto handle() const noexcept -> VkSemaphore { return semaphore_; }

  /**
   * Host-waits until the semaphore reaches at least `value`.
   *
   * No-op when `value == 0`.
   *
   * @param value Timeline value to wait for.
   */
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
