#ifndef VKEXEC_CONTEXT_HPP
#define VKEXEC_CONTEXT_HPP

//! \file
//! Vulkan device context: queues, command-pool cache, and host/completion agents.

#include <vkexec/detail/object_synchronization.hpp>
#include <vkexec/detail/worker_callbacks.hpp>
#include <vkexec/device_procs.hpp>
#include <vkexec/error.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <concepts>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

class queue_guard
{
public:
  ~queue_guard() = default;
  queue_guard(queue_guard const &) = delete;
  auto operator=(queue_guard const &) -> queue_guard & = delete;
  queue_guard(queue_guard &&) noexcept = default;
  auto operator=(queue_guard &&other) noexcept -> queue_guard &
  {
    if (this == &other) { return *this; }
    guard_ = std::move(other.guard_);
    state_ = std::move(other.state_);
    return *this;
  }

private:
  friend class context;
  explicit queue_guard(std::shared_ptr<detail::queue_synchronization_state> state)
    : state_(std::move(state)), guard_(state_->lock())
  {}

  std::shared_ptr<detail::queue_synchronization_state> state_;
  detail::queue_synchronization_state::guard guard_;
};

namespace owned {
  class presenter;
}// namespace owned

class scheduler;
class window;
class context;

namespace detail {
  enum class synchronization_backend : std::uint8_t;
  struct submit_scope;
  [[nodiscard]] auto synchronization_backend_for(context const &ctx) noexcept -> synchronization_backend;
}// namespace detail

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
 * presentation is unused. Direct queue operations by the embedder must use
 * `context::lock_queue` so they share vkexec's queue synchronization. One
 * VkQueue must not be managed by two vkexec contexts at the same time.
 *
 * @see factory::adopt_context
 */
struct context_adopt_info
{
  VkInstance instance{ VK_NULL_HANDLE };
  VkPhysicalDevice physical_device{ VK_NULL_HANDLE };
  VkDevice device{ VK_NULL_HANDLE };
  //! API version negotiated by the embedder.
  std::uint32_t api_version{ VK_API_VERSION_1_0 };
  //! Feature bit enabled when the device was created.
  bool synchronization2_enabled{ false };
  //! Required with the feature bit on devices using the KHR command.
  bool synchronization2_khr_extension_enabled{ false };
  //! Required for the KHR path on Vulkan 1.0 instances.
  bool physical_device_properties2_enabled{ false };
  //! True if VK_KHR_timeline_semaphore was enabled when creating an adopted
  //! Vulkan 1.0/1.1 VkDevice.
  bool timeline_semaphore_khr_extension_enabled{ false };

  VkQueue compute_queue{ VK_NULL_HANDLE };
  std::uint32_t compute_queue_family{ 0 };

  VkQueue graphics_queue{ VK_NULL_HANDLE };
  std::uint32_t graphics_queue_family{ 0 };

  VkQueue present_queue{ VK_NULL_HANDLE };
  std::uint32_t present_queue_family{ 0 };
};

namespace detail {

  struct make_context_factory
  {
    scheduler_options opts;

    [[nodiscard]] auto operator()() const -> result<std::unique_ptr<context>>;
  };

  struct adopt_context_factory
  {
    context_adopt_info info;

    [[nodiscard]] auto operator()() const -> result<std::unique_ptr<context>>;
  };

}// namespace detail

namespace factory {

  /**
   * Creates a compute-only context (no window / swapchain).
   *
   * Completes with `set_error(errc::unsupported)` when device selection or
   * feature negotiation fails.
   *
   * @param opts Validation and extra Vulkan requirements.
   * @return Sender that completes with `unique_ptr<context>` on success.
   */
  struct make_context_t
  {
    [[nodiscard]] auto operator()(scheduler_options const &opts = {}) const
    { return make_sender(detail::make_context_factory{ .opts = opts }); }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_context_t make_context{};

  /**
   * Adopts embedder-owned Vulkan handles without taking destruction ownership.
   *
   * @param info Borrowed instance, device, and queues.
   * @return Sender that completes with `unique_ptr<context>` on success.
   */
  struct adopt_context_t
  {
    [[nodiscard]] auto operator()(context_adopt_info const &info) const
    { return make_sender(detail::adopt_context_factory{ .info = info }); }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr adopt_context_t adopt_context{};

}// namespace factory

/**
 * Owns command pools and host/completion agents. Depending on construction,
 * it either owns or adopts the Vulkan instance/device/queues; query the exact
 * mode with `owns_instance()` and `owns_device()`.
 *
 * Create with `factory::make_context()` for a compute-only device, or
 * `factory::adopt_context()` to wrap embedder-owned handles. Most GPU work is
 * scheduled via `get_scheduler()` and completed with `sync_wait`.
 *
 * Queue operations are serialized per actual VkQueue, including aliases.
 * Descriptor allocation and free are serialized per VkDescriptorPool.
 * Concurrent host access to the same descriptor set, including update and
 * free, remains the caller's responsibility.
 * A command pool and recording of its command buffers require exclusive use.
 * Destruction requires all users and queue operations on the device to have
 * stopped; it may call `vkDeviceWaitIdle` before releasing owned resources.
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
  friend auto detail::synchronization_backend_for(context const &ctx) noexcept -> detail::synchronization_backend;

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
  //! True when this context destroys the Vulkan instance in its destructor.
  [[nodiscard]] auto owns_instance() const noexcept -> bool;
  //! True when this context destroys the Vulkan device in its destructor.
  [[nodiscard]] auto owns_device() const noexcept -> bool;
  //! True when graphics/present queues were configured for swapchain use.
  [[nodiscard]] auto presentation_enabled() const noexcept -> bool;
  //! Effective Vulkan requirements after merging library baselines.
  [[nodiscard]] auto requirements() const noexcept -> vulkan_requirements const &;
  //! Negotiated Vulkan API version for this device.
  [[nodiscard]] auto api_version() const noexcept -> std::uint32_t;
  //! Core device function pointers loaded via `vkGetDeviceProcAddr`.
  [[nodiscard]] auto procs() const noexcept -> device_procs const &;

  /**
   * Allocates a primary command buffer with an exclusive compute-family pool.
   * Caller must return it with `free_command_buffer` after GPU use completes.
   * Recording separate returned buffers may proceed concurrently.
   */
  [[nodiscard]] auto allocate_command_buffer() -> result<VkCommandBuffer>;

  //! Frees `cmd` and returns its exclusive pool to the cache. `cmd` must have
  //! been returned by this context's `allocate_command_buffer()`.
  auto free_command_buffer(VkCommandBuffer cmd) -> void;

  //! Locks the actual queue for direct embedder queue operations.
  [[nodiscard]] auto lock_queue(VkQueue queue) const -> queue_guard;

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
   * Caller-owned `info.fence` requires caller synchronization for concurrent
   * host operations on that fence.
   *
   * @param info Queue submit description.
   * @return Failure if validation, lowering, or Vulkan queue submission fails.
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
    requires stdexec::stoppable_token<std::remove_cvref_t<StopToken>>
             && std::constructible_from<detail::done_fn, Done &&>
  [[nodiscard]] auto enqueue_fence_wait(VkSemaphore semaphore, VkFence fence, StopToken token, Done &&on_done) -> status
  {
    using token_t = std::remove_cvref_t<StopToken>;

    detail::stop_fn stop_requested;
    if constexpr (stdexec::unstoppable_token<token_t>) {
      static_cast<void>(token);
    } else {
      stop_requested = [token = std::move(token)]() noexcept -> bool { return token.stop_requested(); };
    }
    return do_enqueue_fence_wait(
      semaphore, fence, std::move(stop_requested), detail::done_fn(std::forward<Done>(on_done)));
  }

  /**
   * Waits for a caller-owned fence on the completion agent without destroying it.
   *
   * @param fence Fence borrowed from the caller.
   * @param token Stop token polled while waiting.
   * @param on_done Callback receiving optional error and a stopped flag.
   */
  template<class StopToken, class Done>
    requires stdexec::stoppable_token<std::remove_cvref_t<StopToken>>
             && std::constructible_from<detail::done_fn, Done &&>
  [[nodiscard]] auto enqueue_borrowed_fence_wait(VkFence fence, StopToken token, Done &&on_done) -> status
  {
    using token_t = std::remove_cvref_t<StopToken>;

    detail::stop_fn stop_requested;
    if constexpr (stdexec::unstoppable_token<token_t>) {
      static_cast<void>(token);
    } else {
      stop_requested = [token = std::move(token)]() noexcept -> bool { return token.stop_requested(); };
    }
    return do_enqueue_borrowed_fence_wait(
      fence, std::move(stop_requested), detail::done_fn(std::forward<Done>(on_done)));
  }

  /**
   * Runs `task` on the context host agent (schedule completions land here).
   *
   * @param task Callable invoked on the host agent thread.
   */
  template<class Task>
    requires std::constructible_from<detail::host_task_fn, Task &&>
  [[nodiscard]] auto enqueue_host(Task &&task) -> status
  { return do_enqueue_host(detail::host_task_fn(std::forward<Task>(task))); }

  //! Returns the thread id of the host agent, creating it if needed.
  [[nodiscard]] auto host_agent_thread_id() -> std::thread::id;

private:
  friend class owned::presenter;
  friend struct detail::submit_scope;
  friend class detail::descriptor_pool_access;
  // MSVC misparses trailing-return friend decls named like the enclosing class.
  friend struct detail::make_context_factory;
  friend struct detail::adopt_context_factory;

  //! Empty tags for staged construction; only friends can name them.
  struct uninitialized_tag
  {
  };
  struct instance_only_tag
  {
  };

  /**
   * Passkey so friends can `std::make_unique` without exposing public ctors.
   * (`make_unique` is not a friend, so private ctors are inaccessible to it.)
   */
  class factory_access
  {
    factory_access() = default;
    friend struct detail::make_context_factory;
    friend struct detail::adopt_context_factory;
    friend class owned::presenter;
  };

public:
  explicit context(factory_access /*access*/, uninitialized_tag tag) noexcept;
  explicit context(factory_access /*access*/,
    instance_only_tag tag,
    scheduler_options const &opts,
    std::vector<char const *> const &instance_extensions);

private:
  auto init_headless(scheduler_options const &opts) -> status;
  auto init_adopted(context_adopt_info const &info) -> status;
  auto init_common_resources() -> status;
  auto complete_for_surface(VkSurfaceKHR surface) -> status;

  [[nodiscard]] auto acquire_command_pool(std::uint32_t family) -> result<VkCommandPool>;
  auto release_command_pool(std::uint32_t family, VkCommandPool pool) noexcept -> void;
  auto fetch_queues(bool want_present) -> status;
  auto load_device_procs() -> status;
  [[nodiscard]] auto
    do_enqueue_fence_wait(VkSemaphore semaphore, VkFence fence, detail::stop_fn stop_requested, detail::done_fn on_done)
      -> status;
  [[nodiscard]] auto
    do_enqueue_borrowed_fence_wait(VkFence fence, detail::stop_fn stop_requested, detail::done_fn on_done) -> status;
  [[nodiscard]] auto do_enqueue_host(detail::host_task_fn task) -> status;

  struct impl;
  std::unique_ptr<impl> impl_;
};

}// namespace vkexec

#endif// VKEXEC_CONTEXT_HPP
