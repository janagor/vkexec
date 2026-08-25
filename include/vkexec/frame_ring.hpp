#ifndef VKEXEC_FRAME_RING_HPP
#define VKEXEC_FRAME_RING_HPP

#include <vkexec/error.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/timeline_semaphore.hpp>

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vkexec {

/// Wait/signal entries for a frame submit: binary acquire wait, binary render-finished
/// signal, and timeline completion signal (mirrors a typical WSI + timeline ring).
struct frame_ring_submit_sync
{
  std::array<semaphore_submit, 1> waits{};
  std::array<semaphore_submit, 2> signals{};
};

/// Frames-in-flight sync: binary acquire/present semaphores plus one timeline that gates
/// reuse of CPU frame slots and swapchain images (vkgsplat-style ring).
class frame_ring
{
public:
  static constexpr std::size_t k_default_slot_count = 2;

  struct create_info
  {
    std::size_t slot_count{ k_default_slot_count };
    std::size_t image_count{ 0 };
  };

  [[nodiscard]] static auto create(context &ctx, create_info info) -> result<frame_ring>;

  ~frame_ring();

  frame_ring(frame_ring const &) = delete;
  auto operator=(frame_ring const &) -> frame_ring & = delete;

  frame_ring(frame_ring &&other) noexcept;
  auto operator=(frame_ring &&other) noexcept -> frame_ring &;

  /// Recreate per-image binary semaphores after a swapchain recreate. Resets image
  /// completion values; does not reset slot values or the monotonic counter.
  auto resize_images(std::size_t image_count) -> status;

  /// Clear slot/image completion values (e.g. after device idle + swapchain recreate).
  auto reset_completion_tracking() -> void;

  [[nodiscard]] auto slot_count() const noexcept -> std::size_t { return acquire_.size(); }
  [[nodiscard]] auto image_count() const noexcept -> std::size_t { return render_finished_.size(); }

  [[nodiscard]] auto acquire_semaphore(std::size_t slot) const -> result<VkSemaphore>;
  [[nodiscard]] auto render_finished_semaphore(std::size_t image_index) const -> result<VkSemaphore>;
  [[nodiscard]] auto timeline() const noexcept -> timeline_semaphore const & { return timeline_; }
  [[nodiscard]] auto timeline() noexcept -> timeline_semaphore & { return timeline_; }

  /// Host-wait until the GPU finished the previous submit that used this slot.
  [[nodiscard]] auto wait_slot(std::size_t slot) const -> status;

  /// Host-wait until the GPU finished the previous submit that used this image.
  [[nodiscard]] auto wait_image(std::size_t image_index) const -> status;

  /// Next monotonic timeline value for a completing submit.
  [[nodiscard]] auto allocate_signal_value() -> std::uint64_t;

  /// Record that `slot` and `image_index` are gated by `signal_value` after submit.
  auto mark_submitted(std::size_t slot, std::size_t image_index, std::uint64_t signal_value) -> status;

  /// Build wait/signal lists: wait acquire[slot], signal finished[image] + timeline.
  [[nodiscard]] auto make_submit_sync(std::size_t slot,
    std::size_t image_index,
    std::uint64_t signal_value,
    VkPipelineStageFlags acquire_wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT) const
    -> result<frame_ring_submit_sync>;

private:
  frame_ring(context *ctx, timeline_semaphore timeline) noexcept;

  auto destroy() noexcept -> void;
  auto destroy_image_semaphores() noexcept -> void;
  auto create_image_semaphores(std::size_t image_count) -> status;
  [[nodiscard]] auto check_slot(std::size_t slot) const -> status;
  [[nodiscard]] auto check_image(std::size_t image_index) const -> status;

  context *ctx_{ nullptr };
  timeline_semaphore timeline_;
  std::uint64_t next_timeline_value_{ 0 };
  std::vector<VkSemaphore> acquire_;
  std::vector<VkSemaphore> render_finished_;
  std::vector<std::uint64_t> slot_timeline_value_;
  std::vector<std::uint64_t> image_timeline_value_;
};

}// namespace vkexec

#endif// VKEXEC_FRAME_RING_HPP
