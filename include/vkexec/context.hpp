#ifndef VKEXEC_CONTEXT_HPP
#define VKEXEC_CONTEXT_HPP

#include <vkexec/detail/completion_waiter.hpp>
#include <vkexec/detail/host_agent.hpp>
#include <vkexec/pipeline.hpp>

#include <VkBootstrap.h>
#include <stdexec/execution.hpp>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec::edsl {
class trace_scope;
}

namespace vkexec {

class scheduler;
class window;
class pipeline_cache;

/// Options passed when creating a `context` (affects the scheduler from `get_scheduler()`).
struct scheduler_options
{
  bool validation_layers{ false };
};

/// Handles borrowed from an embedder. vkexec never destroys these.
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

class context
{
public:
  /// Compute-only context (no window / swapchain).
  explicit context(scheduler_options opts = {});
  ~context();

  /// Wrap an existing Vulkan device/queues. Returns a context that does not destroy the
  /// instance, device, or an externally supplied VMA allocator. vkexec still owns its
  /// command pool and pipeline cache; create a VMA allocator when `allocator` is null.
  [[nodiscard]] static auto adopt(context_adopt_info const &info) -> std::unique_ptr<context>;

  context(context const &) = delete;
  auto operator=(context const &) -> context & = delete;
  context(context &&) noexcept = delete;
  auto operator=(context &&) noexcept -> context & = delete;

  [[nodiscard]] auto get_scheduler() noexcept -> scheduler;

  [[nodiscard]] auto instance() const noexcept -> VkInstance { return instance_.instance; }
  [[nodiscard]] auto physical_device() const noexcept -> VkPhysicalDevice { return physical_device_.physical_device; }
  [[nodiscard]] auto device() const noexcept -> VkDevice { return device_.device; }
  [[nodiscard]] auto vkb_device() noexcept -> vkb::Device & { return device_; }
  [[nodiscard]] auto vkb_device() const noexcept -> vkb::Device const & { return device_; }
  [[nodiscard]] auto compute_queue() const noexcept -> VkQueue { return compute_queue_; }
  [[nodiscard]] auto graphics_queue() const noexcept -> VkQueue { return graphics_queue_; }
  [[nodiscard]] auto present_queue() const noexcept -> VkQueue { return present_queue_; }
  [[nodiscard]] auto queue_family() const noexcept -> std::uint32_t { return queue_family_; }
  [[nodiscard]] auto graphics_queue_family() const noexcept -> std::uint32_t { return graphics_family_; }
  [[nodiscard]] auto present_queue_family() const noexcept -> std::uint32_t { return present_family_; }
  [[nodiscard]] auto command_pool() const noexcept -> VkCommandPool { return command_pool_; }
  [[nodiscard]] auto allocator() const noexcept -> VmaAllocator { return allocator_; }
  [[nodiscard]] auto presentation_enabled() const noexcept -> bool { return presentation_enabled_; }

  [[nodiscard]] auto get_or_compile(edsl::trace_scope const &trace, std::uint32_t work_count) -> pipeline_resources &;
  [[nodiscard]] auto get_or_create_from_spirv(std::span<std::uint32_t const> spirv, layout_desc const &desc)
    -> pipeline_resources &;

  auto allocate_command_buffer() -> VkCommandBuffer;
  auto free_command_buffer(VkCommandBuffer cmd) -> void;

  /// Serializes command-pool, descriptor-pool, and queue submits across host threads.
  [[nodiscard]] auto lock_host() const -> std::unique_lock<std::mutex>;

  auto submit_and_wait(VkCommandBuffer cmd) -> void;
  auto submit_async(VkCommandBuffer cmd, VkFence *out_fence = nullptr) -> VkSemaphore;

  /// Wait for a submitted fence on the context completion agent, then invoke `on_done`.
  /// Always waits for the GPU and destroys `semaphore`/`fence` before the callback.
  /// Choose `set_stopped` / `set_value` / `set_error` only after reclaiming cmd/descriptor loans too.
  template<class StopToken, class Done>
  auto enqueue_fence_wait(VkSemaphore semaphore, VkFence fence, StopToken token, Done &&on_done) -> void
  {
    detail::completion_waiter::stop_fn stop_requested;
    if constexpr (!stdexec::unstoppable_token<std::remove_cvref_t<StopToken>>) {
      stop_requested = [token]() -> bool { return token.stop_requested(); };
    }
    ensure_completion_waiter().enqueue(semaphore, fence, std::move(stop_requested), std::forward<Done>(on_done));
  }

  /// Wait for a caller-owned fence on the completion agent (does not destroy the fence).
  template<class StopToken, class Done>
  auto enqueue_borrowed_fence_wait(VkFence fence, StopToken token, Done &&on_done) -> void
  {
    detail::completion_waiter::stop_fn stop_requested;
    if constexpr (!stdexec::unstoppable_token<std::remove_cvref_t<StopToken>>) {
      stop_requested = [token]() -> bool { return token.stop_requested(); };
    }
    ensure_completion_waiter().enqueue_borrowed(fence, std::move(stop_requested), std::forward<Done>(on_done));
  }

  /// Run `task` on the context host agent (schedule completions land here).
  template<class Task> auto enqueue_host(Task &&task) -> void
  { ensure_host_agent().enqueue(detail::host_agent::task_fn{ std::forward<Task>(task) }); }

  [[nodiscard]] auto host_agent_thread_id() -> std::thread::id { return ensure_host_agent().thread_id(); }

private:
  friend class pipeline_cache;
  friend class window;
  template<typename T> friend class buffer;

  struct instance_only_tag
  {
  };
  explicit context(instance_only_tag tag, scheduler_options opts, std::vector<char const *> const &instance_extensions);
  explicit context(context_adopt_info const &info);
  auto complete_for_surface(VkSurfaceKHR surface) -> void;

  auto create_command_pool() -> void;
  auto create_allocator() -> void;
  auto fetch_queues(bool want_present) -> void;
  auto ensure_completion_waiter() -> detail::completion_waiter &;
  auto ensure_host_agent() -> detail::host_agent &;

  vkb::Instance instance_{};
  vkb::PhysicalDevice physical_device_{};
  vkb::Device device_{};
  VmaAllocator allocator_{ VK_NULL_HANDLE };
  VkQueue compute_queue_{ VK_NULL_HANDLE };
  VkQueue graphics_queue_{ VK_NULL_HANDLE };
  VkQueue present_queue_{ VK_NULL_HANDLE };
  std::uint32_t queue_family_{ 0 };
  std::uint32_t graphics_family_{ 0 };
  std::uint32_t present_family_{ 0 };
  VkCommandPool command_pool_{ VK_NULL_HANDLE };
  std::unique_ptr<pipeline_cache> pipeline_cache_;
  std::unique_ptr<detail::completion_waiter> completion_waiter_;
  std::unique_ptr<detail::host_agent> host_agent_;
  mutable std::mutex host_mutex_;
  bool presentation_enabled_{ false };
  bool has_instance_{ false };
  bool has_device_{ false };
  bool owns_instance_{ false };
  bool owns_device_{ false };
  bool owns_allocator_{ false };
};

}// namespace vkexec

#endif// VKEXEC_CONTEXT_HPP
