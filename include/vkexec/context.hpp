#ifndef VKEXEC_CONTEXT_HPP
#define VKEXEC_CONTEXT_HPP

//! \file
//! Vulkan device context: queues, VMA, command pool, and host/completion agents.

#include <vkexec/device_procs.hpp>
#include <vkexec/error.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <VkBootstrap.h>
#include <stdexec/execution.hpp>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

class scheduler;
class window;

/**
 * Options passed when creating a `context`.
 *
 * Library baseline requirements are merged with `requirements` at create time;
 * user requirements never remove library floors.
 *
 * @see factory::make_context, vulkan_requirements
 */
struct scheduler_options
{
  //! When true, enables Vulkan validation layers if available.
  bool validation_layers{ false };
  //! Extra instance/device extensions and features requested by the caller.
  vulkan_requirements requirements{};
};

/**
 * Handles borrowed from an embedder when adopting an existing Vulkan device.
 *
 * vkexec never destroys these objects. Queues and family indices must be valid
 * for the provided device; optional graphics/present queues may be null when
 * presentation is unused.
 *
 * @see factory::adopt_context
 */
struct context_adopt_info
{
  VkInstance instance{ VK_NULL_HANDLE };
  VkPhysicalDevice physical_device{ VK_NULL_HANDLE };
  VkDevice device{ VK_NULL_HANDLE };
  VmaAllocator allocator{ VK_NULL_HANDLE };

  VkQueue compute_queue{ VK_NULL_HANDLE };
  std::uint32_t compute_queue_family{ 0 };

  VkQueue graphics_queue{ VK_NULL_HANDLE };
  std::uint32_t graphics_queue_family{ 0 };

  VkQueue present_queue{ VK_NULL_HANDLE };
  std::uint32_t present_queue_family{ 0 };
};

class context;

namespace factory {

  struct make_context_t
  {
    [[nodiscard]] auto operator()(scheduler_options const &opts = {}) const -> sender<std::unique_ptr<context>>;
  };

  inline constexpr make_context_t make_context{};

  /**
   * Creates a compute-only context (no window / swapchain).
   *
   * Completes with `set_error(errc::unsupported)` when device selection or
   * feature negotiation fails.
   *
   * @param opts Validation and extra Vulkan requirements.
   * @return Sender that completes with `unique_ptr<context>` on success.
   */
  /**
   * Adopts embedder-owned Vulkan handles without taking destruction ownership.
   *
   * @param info Borrowed instance, device, allocator, and queues.
   * @return Sender that completes with `unique_ptr<context>` on success.
   */
  struct adopt_context_t
  {
    [[nodiscard]] auto operator()(context_adopt_info const &info) const -> sender<std::unique_ptr<context>>;
  };

  inline constexpr adopt_context_t adopt_context{};

}// namespace factory

/**
 * Owns (or adopts) the Vulkan instance/device, queues, VMA allocator, and host
 * agents used by vkexec senders.
 *
 * Create with `factory::make_context()` for a compute-only device, or
 * `factory::adopt_context()` to wrap embedder-owned handles. Most GPU work is
 * scheduled via `get_scheduler()` and completed with `sync_wait`.
 *
 * Thread safety: command-pool, descriptor-pool, and queue submits must be
 * serialized with `lock_host()` (or use the provided sender adaptors).
 *
 * ~~~~~~~~~~~{.cpp}
 * auto ctx = vkexec::sync_wait_value(vkexec::factory::make_context());
 * auto sched = ctx->get_scheduler();
 * ~~~~~~~~~~~
 *
 * @see factory::make_context, factory::adopt_context, scheduler, sync_wait, vulkan_requirements
 */
class context
{
public:
  ~context();

  context(context const &) = delete;
  auto operator=(context const &) -> context & = delete;
  context(context &&) noexcept = delete;
  auto operator=(context &&) noexcept -> context & = delete;

  //! Returns a scheduler whose completions run on this context's host agent.
  [[nodiscard]] auto get_scheduler() noexcept -> scheduler;

  //! Borrowed Vulkan instance handle.
  [[nodiscard]] auto instance() const noexcept -> VkInstance;
  //! Borrowed physical device handle.
  [[nodiscard]] auto physical_device() const noexcept -> VkPhysicalDevice;
  //! Borrowed logical device handle.
  [[nodiscard]] auto device() const noexcept -> VkDevice;
  //! Mutable VkBootstrap device wrapper.
  [[nodiscard]] auto vkb_device() noexcept -> vkb::Device &;
  //! Const VkBootstrap device wrapper.
  [[nodiscard]] auto vkb_device() const noexcept -> vkb::Device const &;
  //! Compute queue used for dispatch and most submits.
  [[nodiscard]] auto compute_queue() const noexcept -> VkQueue;
  //! Graphics queue when presentation or graphics work is enabled; may be null.
  [[nodiscard]] auto graphics_queue() const noexcept -> VkQueue;
  //! Present queue when swapchain presentation is enabled; may be null.
  [[nodiscard]] auto present_queue() const noexcept -> VkQueue;
  //! Queue family index for `compute_queue()`.
  [[nodiscard]] auto queue_family() const noexcept -> std::uint32_t;
  //! Queue family index for `graphics_queue()`.
  [[nodiscard]] auto graphics_queue_family() const noexcept -> std::uint32_t;
  //! Queue family index for `present_queue()`.
  [[nodiscard]] auto present_queue_family() const noexcept -> std::uint32_t;
  //! Shared command pool for transient primary command buffers.
  [[nodiscard]] auto command_pool() const noexcept -> VkCommandPool;
  //! VMA allocator used for buffers and images.
  [[nodiscard]] auto allocator() const noexcept -> VmaAllocator;
  //! True when graphics/present queues were configured for swapchain use.
  [[nodiscard]] auto presentation_enabled() const noexcept -> bool;
  //! Effective Vulkan requirements after merging library baselines.
  [[nodiscard]] auto requirements() const noexcept -> vulkan_requirements const &;
  //! Negotiated Vulkan API version for this device.
  [[nodiscard]] auto api_version() const noexcept -> std::uint32_t;
  //! Core device function pointers loaded via `vkGetDeviceProcAddr`.
  [[nodiscard]] auto procs() const noexcept -> device_procs const &;

  /**
   * Allocates a primary command buffer from the context command pool.
   *
   * Caller must return it with `free_command_buffer`. Hold `lock_host()` across
   * allocate/record/submit when sharing the context across threads.
   */
  [[nodiscard]] auto allocate_command_buffer() -> result<VkCommandBuffer>;

  //! Returns `cmd` to the context command pool.
  auto free_command_buffer(VkCommandBuffer cmd) -> void;

  /**
   * Locks the host mutex that serializes command-pool, descriptor-pool, and
   * queue submits across host threads.
   */
  [[nodiscard]] auto lock_host() const -> std::unique_lock<std::mutex>;

  /**
   * Submits `cmd` and blocks until the GPU finishes.
   *
   * @param cmd Primary command buffer previously begun and ended by the caller.
   */
  [[nodiscard]] auto submit_and_wait(VkCommandBuffer cmd) -> status;

  /**
   * Submits `cmd` without blocking; optionally returns a binary semaphore and fence.
   *
   * When non-null, `out_semaphore` and `out_fence` receive newly created handles
   * owned by the caller (or by a later `enqueue_fence_wait` path).
   *
   * @param cmd Recorded primary command buffer.
   * @param out_semaphore Optional destination for a signalled binary semaphore.
   * @param out_fence Optional destination for a fence signalled on completion.
   */
  [[nodiscard]] auto submit_async(VkCommandBuffer cmd, VkSemaphore *out_semaphore, VkFence *out_fence = nullptr)
    -> status;

  /**
   * Submits work described by `info` (command buffers plus optional wait/signal
   * binary or timeline semaphores).
   *
   * @param info Queue submit description.
   * @return Failure if `vkQueueSubmit` fails.
   */
  [[nodiscard]] auto submit(queue_submit const &info) const -> status;

  /**
   * Waits for a submitted fence on the context completion agent, then invokes `on_done`.
   *
   * Always waits for the GPU and destroys `semaphore`/`fence` before the callback.
   * Choose `set_stopped` / `set_value` / `set_error` only after reclaiming command
   * buffer and descriptor loans too.
   *
   * @param semaphore Binary semaphore destroyed after the wait (may be null).
   * @param fence Fence destroyed after the wait.
   * @param token Stop token polled while waiting.
   * @param on_done Callback receiving optional error and a stopped flag.
   */
  template<class StopToken, class Done>
  [[nodiscard]] auto enqueue_fence_wait(VkSemaphore semaphore, VkFence fence, StopToken token, Done &&on_done) -> status
  {
    std::function<bool()> stop_requested;
    if constexpr (!stdexec::unstoppable_token<std::remove_cvref_t<StopToken>>) {
      auto state = std::make_shared<std::remove_cvref_t<StopToken>>(std::move(token));
      stop_requested = [state]() -> bool { return state->stop_requested(); };
    }
    auto done = std::make_shared<std::remove_cvref_t<Done>>(std::forward<Done>(on_done));
    return do_enqueue_fence_wait(
      semaphore, fence, std::move(stop_requested), [done](std::optional<error> failure, bool stopped) mutable -> void {
        std::invoke(*done, std::move(failure), stopped);
      });
  }

  /**
   * Waits for a caller-owned fence on the completion agent without destroying it.
   *
   * @param fence Fence borrowed from the caller.
   * @param token Stop token polled while waiting.
   * @param on_done Callback receiving optional error and a stopped flag.
   */
  template<class StopToken, class Done>
  [[nodiscard]] auto enqueue_borrowed_fence_wait(VkFence fence, StopToken token, Done &&on_done) -> status
  {
    std::function<bool()> stop_requested;
    if constexpr (!stdexec::unstoppable_token<std::remove_cvref_t<StopToken>>) {
      auto state = std::make_shared<std::remove_cvref_t<StopToken>>(std::move(token));
      stop_requested = [state]() -> bool { return state->stop_requested(); };
    }
    auto done = std::make_shared<std::remove_cvref_t<Done>>(std::forward<Done>(on_done));
    return do_enqueue_borrowed_fence_wait(
      fence, std::move(stop_requested), [done](std::optional<error> failure, bool stopped) mutable -> void {
        std::invoke(*done, std::move(failure), stopped);
      });
  }

  /**
   * Runs `task` on the context host agent (schedule completions land here).
   *
   * @param task Callable invoked on the host agent thread.
   */
  template<class Task> [[nodiscard]] auto enqueue_host(Task &&task) -> status
  {
    auto state = std::make_shared<std::remove_cvref_t<Task>>(std::forward<Task>(task));
    return do_enqueue_host([state]() mutable -> void { std::invoke(*state); });
  }

  //! Returns the thread id of the host agent, creating it if needed.
  [[nodiscard]] auto host_agent_thread_id() -> std::thread::id;

private:
  friend class presenter;
  template<typename T> friend class buffer;
  // MSVC misparses trailing-return friend decls named like the enclosing class.
  friend struct factory::make_context_t;
  friend struct factory::adopt_context_t;

  struct uninitialized_tag
  {
  };
  struct instance_only_tag
  {
  };

  explicit context(uninitialized_tag tag) noexcept;
  explicit context(instance_only_tag tag,
    scheduler_options const &opts,
    std::vector<char const *> const &instance_extensions);

  auto init_headless(scheduler_options const &opts) -> status;
  auto init_adopted(context_adopt_info const &info) -> status;
  auto init_common_resources() -> status;
  auto complete_for_surface(VkSurfaceKHR surface) -> status;

  auto create_command_pool() -> status;
  auto create_allocator() -> status;
  auto fetch_queues(bool want_present) -> status;
  auto load_device_procs() -> void;
  [[nodiscard]] auto do_enqueue_fence_wait(VkSemaphore semaphore,
    VkFence fence,
    std::function<bool()> stop_requested,
    std::function<void(std::optional<error>, bool)> on_done) -> status;
  [[nodiscard]] auto do_enqueue_borrowed_fence_wait(VkFence fence,
    std::function<bool()> stop_requested,
    std::function<void(std::optional<error>, bool)> on_done) -> status;
  [[nodiscard]] auto do_enqueue_host(std::function<void()> task) -> status;

  struct impl;
  std::unique_ptr<impl> impl_;
};

}// namespace vkexec

#endif// VKEXEC_CONTEXT_HPP
