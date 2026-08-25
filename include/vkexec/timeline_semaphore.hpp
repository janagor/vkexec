#ifndef VKEXEC_TIMELINE_SEMAPHORE_HPP
#define VKEXEC_TIMELINE_SEMAPHORE_HPP

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>

namespace vkexec {

/// RAII timeline semaphore (`VK_SEMAPHORE_TYPE_TIMELINE`).
class timeline_semaphore
{
public:
  [[nodiscard]] static auto create(context &ctx, std::uint64_t initial_value = 0) -> timeline_semaphore;

  ~timeline_semaphore();

  timeline_semaphore(timeline_semaphore const &) = delete;
  auto operator=(timeline_semaphore const &) -> timeline_semaphore & = delete;

  timeline_semaphore(timeline_semaphore &&other) noexcept;
  auto operator=(timeline_semaphore &&other) noexcept -> timeline_semaphore &;

  [[nodiscard]] auto handle() const noexcept -> VkSemaphore { return semaphore_; }

  /// Host wait until the semaphore reaches at least `value` (no-op when value == 0).
  auto wait(std::uint64_t value) const -> void;

private:
  timeline_semaphore(context *ctx, VkSemaphore semaphore) noexcept;
  auto destroy() noexcept -> void;

  context *ctx_{ nullptr };
  VkSemaphore semaphore_{ VK_NULL_HANDLE };
};

}// namespace vkexec

#endif// VKEXEC_TIMELINE_SEMAPHORE_HPP
