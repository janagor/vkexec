#ifndef VKEXEC_CONTEXT_HPP
#define VKEXEC_CONTEXT_HPP

#include <vkexec/device_procs.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/queue_submit.hpp>
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

namespace vkexec::edsl {
class trace_scope;
}

namespace vkexec {

class scheduler;
class window;
class pipeline_cache;

namespace detail {
  class completion_waiter;
  class host_agent;
}// namespace detail

/// Options passed when creating a `context`.
struct scheduler_options
{
  bool validation_layers{ false };
  vulkan_requirements requirements{};
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
  ~context();

  /// Compute-only context (no window / swapchain). Returns `errc::unsupported` when device
  /// selection or feature negotiation fails.
  [[nodiscard]] static auto create(scheduler_options const &opts = {}) -> result<std::unique_ptr<context>>;

  /// Wrap an existing Vulkan device/queues. Returns a context that does not destroy the
  /// instance, device, or an externally supplied VMA allocator. vkexec still owns its
  /// command pool and pipeline cache; create a VMA allocator when `allocator` is null.
  [[nodiscard]] static auto adopt(context_adopt_info const &info) -> result<std::unique_ptr<context>>;

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
  [[nodiscard]] auto requirements() const noexcept -> vulkan_requirements const & { return requirements_; }
  [[nodiscard]] auto api_version() const noexcept -> std::uint32_t { return api_version_; }
  [[nodiscard]] auto procs() const noexcept -> device_procs const & { return procs_; }

  [[nodiscard]] auto get_or_compile(edsl::trace_scope const &trace, std::uint32_t work_count)
    -> result<std::reference_wrapper<pipeline_resources>>;
  [[nodiscard]] auto get_or_create_from_spirv(std::span<std::uint32_t const> spirv, layout_desc const &desc)
    -> result<std::reference_wrapper<pipeline_resources>>;

  [[nodiscard]] auto allocate_command_buffer() -> result<VkCommandBuffer>;
  auto free_command_buffer(VkCommandBuffer cmd) -> void;

  /// Serializes command-pool, descriptor-pool, and queue submits across host threads.
  [[nodiscard]] auto lock_host() const -> std::unique_lock<std::mutex>;

  [[nodiscard]] auto submit_and_wait(VkCommandBuffer cmd) -> status;
  [[nodiscard]] auto submit_async(VkCommandBuffer cmd, VkSemaphore *out_semaphore, VkFence *out_fence = nullptr)
    -> status;
  /// Submit with optional binary/timeline wait and signal semaphores.
  [[nodiscard]] auto submit(queue_submit const &info) const -> status;

  /// Wait for a submitted fence on the context completion agent, then invoke `on_done`.
  /// Always waits for the GPU and destroys `semaphore`/`fence` before the callback.
  /// Choose `set_stopped` / `set_value` / `set_error` only after reclaiming cmd/descriptor loans too.
  template<class StopToken, class Done>
  [[nodiscard]] auto enqueue_fence_wait(VkSemaphore semaphore, VkFence fence, StopToken token, Done &&on_done) -> status
  {
    std::move_only_function<bool()> stop_requested;
    if constexpr (!stdexec::unstoppable_token<std::remove_cvref_t<StopToken>>) {
      stop_requested = [token]() -> bool { return token.stop_requested(); };
    }
    return do_enqueue_fence_wait(semaphore,
      fence,
      std::move(stop_requested),
      std::move_only_function<void(std::optional<error>, bool)>{ std::forward<Done>(on_done) });
  }

  /// Wait for a caller-owned fence on the completion agent (does not destroy the fence).
  template<class StopToken, class Done>
  [[nodiscard]] auto enqueue_borrowed_fence_wait(VkFence fence, StopToken token, Done &&on_done) -> status
  {
    std::move_only_function<bool()> stop_requested;
    if constexpr (!stdexec::unstoppable_token<std::remove_cvref_t<StopToken>>) {
      stop_requested = [token]() -> bool { return token.stop_requested(); };
    }
    return do_enqueue_borrowed_fence_wait(fence,
      std::move(stop_requested),
      std::move_only_function<void(std::optional<error>, bool)>{ std::forward<Done>(on_done) });
  }

  /// Run `task` on the context host agent (schedule completions land here).
  template<class Task> [[nodiscard]] auto enqueue_host(Task &&task) -> status
  { return do_enqueue_host(std::move_only_function<void()>{ std::forward<Task>(task) }); }

  [[nodiscard]] auto host_agent_thread_id() -> std::thread::id;

private:
  friend class pipeline_cache;
  friend class window;
  template<typename T> friend class buffer;

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
  auto ensure_completion_waiter() -> result<detail::completion_waiter *>;
  auto ensure_host_agent() -> result<detail::host_agent *>;
  [[nodiscard]] auto do_enqueue_fence_wait(VkSemaphore semaphore,
    VkFence fence,
    std::move_only_function<bool()> stop_requested,
    std::move_only_function<void(std::optional<error>, bool)> on_done) -> status;
  [[nodiscard]] auto do_enqueue_borrowed_fence_wait(VkFence fence,
    std::move_only_function<bool()> stop_requested,
    std::move_only_function<void(std::optional<error>, bool)> on_done) -> status;
  [[nodiscard]] auto do_enqueue_host(std::move_only_function<void()> task) -> status;

  vulkan_requirements requirements_{};
  std::uint32_t api_version_{ VK_API_VERSION_1_0 };
  device_procs procs_{};
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
