#ifndef VKEXEC_COMPLETION_WAITER_HPP
#define VKEXEC_COMPLETION_WAITER_HPP

#include <vkexec/detail/move_only_function.hpp>
#include <vkexec/result.hpp>
#include <vkexec/error.hpp>

#include <vulkan/vulkan.h>

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace vkexec::detail {

/// Context-owned agent that waits on GPU fences and delivers completions.
/// Multiple outstanding fences are polled together so overlapped submits stay concurrent.
class completion_waiter
{
public:
  using done_fn = move_only_function<void(std::optional<error> failure, bool stopped)>;
  using stop_fn = move_only_function<bool()>;

  completion_waiter(VkDevice device, VkQueue fallback_queue);
  ~completion_waiter();

  completion_waiter(completion_waiter const &) = delete;
  auto operator=(completion_waiter const &) -> completion_waiter & = delete;
  completion_waiter(completion_waiter &&) = delete;
  auto operator=(completion_waiter &&) -> completion_waiter & = delete;

  /// Always waits for the GPU and destroys `semaphore`/`fence` before invoking `on_done`.
  auto enqueue(VkSemaphore semaphore, VkFence fence, stop_fn stop_requested, done_fn on_done) -> status;

  /// Wait for a caller-owned fence; never destroys it. Used for window frame fences.
  auto enqueue_borrowed(VkFence fence, stop_fn stop_requested, done_fn on_done) -> status;

  /// Drain outstanding waits and join the agent thread. Safe to call once.
  auto shutdown() -> void;

private:
  struct job
  {
    VkSemaphore semaphore{ VK_NULL_HANDLE };
    VkFence fence{ VK_NULL_HANDLE };
    stop_fn stop_requested;
    done_fn on_done;
    bool stop_seen{ false };
    bool destroy_sync{ true };
  };

  auto run() -> void;
  auto finish_job(job item, std::optional<error> const &failure) -> void;
  auto finish_all(std::vector<job> &jobs, std::optional<error> const &failure) -> void;
  [[nodiscard]] auto wait_any_fence(std::vector<VkFence> const &fences) -> std::optional<error>;
  auto complete_without_fences(std::vector<job> &jobs) -> void;
  auto reap_ready_jobs(std::vector<job> &jobs) -> void;

  static auto poll_stop_flags(std::vector<job> &jobs) -> void;
  static auto collect_fences(std::vector<job> const &jobs, std::vector<VkFence> &fences) -> void;

  VkDevice device_{ VK_NULL_HANDLE };
  VkQueue fallback_queue_{ VK_NULL_HANDLE };
  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<job> pending_;
  bool shutting_down_{ false };
  std::jthread thread_;
};

}// namespace vkexec::detail

#endif// VKEXEC_COMPLETION_WAITER_HPP
