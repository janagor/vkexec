#ifndef VKEXEC_EXTENSIONS_TIMELINE_SEMAPHORE_FRAME_RING_HPP
#define VKEXEC_EXTENSIONS_TIMELINE_SEMAPHORE_FRAME_RING_HPP

//! \file
//! Frames-in-flight sync: binary acquire/present semaphores plus a timeline ring.

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec_extensions/timeline_semaphore/timeline_semaphore.hpp>

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vkexec {

/**
 * Wait/signal entries for a frame submit.
 *
 * Contains a binary acquire wait, a binary render-finished signal, and a
 * timeline completion signal (typical WSI + timeline ring).
 *
 * @see frame_ring::make_submit_sync, queue_submit
 */
struct frame_ring_submit_sync
{
  std::array<semaphore_submit, 1> waits{};
  std::array<semaphore_submit, 2> signals{};
};

//! Default CPU frame-slot count for `frame_ring_create_info`.
constexpr std::size_t k_default_frame_ring_slot_count = 2;

//! Creation parameters for `factory::make_frame_ring`: CPU slots and initial swapchain image count.
struct frame_ring_create_info
{
  std::size_t slot_count{ k_default_frame_ring_slot_count };
  std::size_t image_count{ 0 };
};

namespace owned {
  class frame_ring;
}// namespace owned

namespace detail {

  struct make_frame_ring_factory
  {
    context *ctx;
    frame_ring_create_info info;

    [[nodiscard]] auto operator()() const -> result<owned::frame_ring>;
  };

}// namespace detail

namespace factory {

  struct make_frame_ring_t
  {

    /**
     * Creates a frame ring on `ctx`.
     *
     * @param ctx Context that owns the device (timeline + binary semaphores).
     * @param info Slot count and initial image count.
     */
    [[nodiscard]] auto operator()(context &ctx, frame_ring_create_info info) const
    { return make_sender(detail::make_frame_ring_factory{ .ctx = &ctx, .info = info }); }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_frame_ring_t make_frame_ring{};

}// namespace factory

/**
 * Frames-in-flight sync: per-slot acquire semaphores, per-image present
 * semaphores, and one timeline that gates reuse of CPU slots and swapchain images.
 *
 * Typical frames-in-flight flow: wait the slot, acquire, record, submit
 * with `make_submit_sync`, then `mark_submitted`.
 *
 * ~~~~~~~~~~~{.cpp}
 * auto ring = vkexec::sync_wait_value(vkexec::factory::make_frame_ring(*ctx, {.slot_count = 2, .image_count = n}));
 * auto value = ring.allocate_signal_value();
 * auto sync = *ring.make_submit_sync(slot, image, value);
 * // ... queue_submit with sync.waits / sync.signals ...
 * ring.mark_submitted(slot, image, value);
 * ~~~~~~~~~~~
 *
 * @see timeline_semaphore, frame_present, factory::make_frame_ring
 */
namespace owned {

  class frame_ring
  {
  public:
    static constexpr std::size_t k_default_slot_count = k_default_frame_ring_slot_count;

    //! @see frame_ring_create_info
    using create_info = frame_ring_create_info;

    ~frame_ring();

    frame_ring(frame_ring const &) = delete;
    auto operator=(frame_ring const &) -> frame_ring & = delete;

    frame_ring(frame_ring &&other) noexcept;
    auto operator=(frame_ring &&other) noexcept -> frame_ring &;

    /**
     * Recreates per-image binary semaphores after a swapchain recreate.
     *
     * Resets image completion values; does not reset slot values or the monotonic counter.
     *
     * @param new_image_count New swapchain image count.
     */
    auto resize_images(std::size_t new_image_count) -> status;

    /**
     * Clears slot/image completion gates without rewinding the timeline sequence.
     *
     * Call only after device idle, typically during swapchain recreation. A Vulkan
     * timeline semaphore cannot be reset, so future signal values remain strictly
     * greater than every value previously allocated by this ring.
     */
    auto reset_completion_tracking() -> void;

    //! Number of CPU frame slots (acquire semaphores).
    [[nodiscard]] auto slot_count() const noexcept -> std::size_t { return acquire_.size(); }
    //! Number of per-image render-finished semaphores.
    [[nodiscard]] auto image_count() const noexcept -> std::size_t { return render_finished_.size(); }

    //! Returns the binary acquire semaphore for `slot`.
    [[nodiscard]] auto acquire_semaphore(std::size_t slot) const -> result<VkSemaphore>;
    //! Returns the binary render-finished semaphore for `image_index`.
    [[nodiscard]] auto render_finished_semaphore(std::size_t image_index) const -> result<VkSemaphore>;
    //! Const reference to the shared timeline semaphore.
    [[nodiscard]] auto timeline() const noexcept -> timeline_semaphore const & { return timeline_; }
    //! Mutable reference to the shared timeline semaphore.
    [[nodiscard]] auto timeline() noexcept -> timeline_semaphore & { return timeline_; }

    //! Host-waits until the GPU finished the previous submit that used this slot.
    [[nodiscard]] auto wait_slot(std::size_t slot) const -> status;

    //! Host-waits until the GPU finished the previous submit that used this image.
    [[nodiscard]] auto wait_image(std::size_t image_index) const -> status;

    //! Allocates the next monotonic timeline value for a completing submit.
    [[nodiscard]] auto allocate_signal_value() -> std::uint64_t;

    /**
     * Records that `slot` and `image_index` are gated by `signal_value` after submit.
     *
     * Subsequent `wait_slot` / `wait_image` wait for this timeline value.
     */
    auto mark_submitted(std::size_t slot, std::size_t image_index, std::uint64_t signal_value) -> status;

    /**
     * Builds wait/signal lists: wait acquire[`slot`], signal finished[`image`] + timeline.
     *
     * @param slot CPU frame slot index.
     * @param image_index Swapchain image index.
     * @param signal_value Timeline value allocated via `allocate_signal_value`.
     * @param acquire_wait_stage Pipeline stage for the acquire wait.
     */
    [[nodiscard]] auto make_submit_sync(std::size_t slot,
      std::size_t image_index,
      std::uint64_t signal_value,
      VkPipelineStageFlags acquire_wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT) const
      -> result<frame_ring_submit_sync>;

  private:
    friend struct detail::make_frame_ring_factory;

    frame_ring(context *ctx, timeline_semaphore timeline_sem) noexcept;

    auto destroy() noexcept -> void;
    auto destroy_image_semaphores() noexcept -> void;
    auto create_image_semaphores(std::size_t new_image_count) -> status;
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

}// namespace owned

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_TIMELINE_SEMAPHORE_FRAME_RING_HPP
