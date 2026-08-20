#include <vkexec/detail/completion_waiter.hpp>
#include <vkexec/detail/config.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <exception>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vkexec::detail {
namespace {

  constexpr std::uint64_t k_poll_timeout_ns = 1'000'000ULL;// 1 ms

  auto destroy_sync(VkDevice device, VkSemaphore semaphore, VkFence fence) noexcept -> void
  {
    if (semaphore != VK_NULL_HANDLE) { vkDestroySemaphore(device, semaphore, nullptr); }
    if (fence != VK_NULL_HANDLE) { vkDestroyFence(device, fence, nullptr); }
  }

  auto make_runtime_error(char const *what) -> std::exception_ptr
  {
    return std::make_exception_ptr(std::runtime_error(what));
  }

}// namespace

completion_waiter::completion_waiter(VkDevice device, VkQueue fallback_queue)
  : device_(device), fallback_queue_(fallback_queue), thread_([this]() -> void { run(); })
{}

completion_waiter::~completion_waiter() { shutdown(); }

auto completion_waiter::enqueue(VkSemaphore semaphore, VkFence fence, stop_fn stop_requested, done_fn on_done) -> void
{
  {
    std::scoped_lock const lock(mutex_);
    if (shutting_down_) {
      destroy_sync(device_, semaphore, fence);
      VKEXEC_THROW(std::runtime_error("completion_waiter enqueue after shutdown"));
    }
    pending_.push_back(job{
      .semaphore = semaphore,
      .fence = fence,
      .stop_requested = std::move(stop_requested),
      .on_done = std::move(on_done),
      .stop_seen = false,
    });
  }
  cv_.notify_one();
}

auto completion_waiter::shutdown() -> void
{
  {
    std::scoped_lock const lock(mutex_);
    if (shutting_down_) { return; }
    shutting_down_ = true;
  }
  cv_.notify_all();
  // jthread joins on destruction; reset so drain finishes before device teardown.
  thread_ = std::jthread{};
}

auto completion_waiter::finish_job(job item, std::exception_ptr error) -> void
{
  destroy_sync(device_, item.semaphore, item.fence);
  if (item.on_done) { item.on_done(std::move(error), item.stop_seen); }
}

auto completion_waiter::finish_all(std::vector<job> &jobs, std::exception_ptr const &error) -> void
{
  for (job &item : jobs) { finish_job(std::move(item), error); }
  jobs.clear();
}

auto completion_waiter::poll_stop_flags(std::vector<job> &jobs) -> void
{
  for (job &item : jobs) {
    if (item.stop_requested) { item.stop_seen = item.stop_seen || item.stop_requested(); }
  }
}

auto completion_waiter::collect_fences(std::vector<job> const &jobs, std::vector<VkFence> &fences) -> void
{
  fences.clear();
  fences.reserve(jobs.size());
  for (job const &item : jobs) {
    if (item.fence != VK_NULL_HANDLE) { fences.push_back(item.fence); }
  }
}

auto completion_waiter::wait_any_fence(std::vector<VkFence> const &fences) -> std::exception_ptr
{
  if (fences.empty()) { return nullptr; }
  VkResult const wait_result =
    vkWaitForFences(device_, static_cast<std::uint32_t>(fences.size()), fences.data(), VK_FALSE, k_poll_timeout_ns);
  if (wait_result == VK_SUCCESS || wait_result == VK_TIMEOUT) { return nullptr; }
  return make_runtime_error("vkWaitForFences failed");
}

auto completion_waiter::complete_without_fences(std::vector<job> &jobs) -> void
{
  if (vkQueueWaitIdle(fallback_queue_) != VK_SUCCESS) {
    finish_all(jobs, make_runtime_error("vkQueueWaitIdle failed"));
    return;
  }
  finish_all(jobs, nullptr);
}

auto completion_waiter::reap_ready_jobs(std::vector<job> &jobs) -> void
{
  std::vector<job> still_waiting;
  still_waiting.reserve(jobs.size());

  for (job &item : jobs) {
    if (item.fence == VK_NULL_HANDLE) {
      finish_job(std::move(item), nullptr);
      continue;
    }

    VkResult const status = vkGetFenceStatus(device_, item.fence);
    if (status == VK_SUCCESS) {
      finish_job(std::move(item), nullptr);
      continue;
    }
    if (status == VK_NOT_READY) {
      still_waiting.push_back(std::move(item));
      continue;
    }
    finish_job(std::move(item), make_runtime_error("vkGetFenceStatus failed"));
  }

  jobs = std::move(still_waiting);
}

auto completion_waiter::run() -> void
{
  std::vector<job> active;
  std::vector<VkFence> fences;

  while (true) {
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [this, &active]() -> bool { return shutting_down_ || !pending_.empty() || !active.empty(); });
      if (!pending_.empty()) {
        active.insert(active.end(),
          std::make_move_iterator(pending_.begin()),
          std::make_move_iterator(pending_.end()));
        pending_.clear();
      }
      if (shutting_down_ && active.empty()) { return; }
    }

    poll_stop_flags(active);
    collect_fences(active, fences);

    if (fences.empty()) {
      if (!active.empty()) { complete_without_fences(active); }
      continue;
    }

    if (std::exception_ptr const wait_error = wait_any_fence(fences)) {
      finish_all(active, wait_error);
      continue;
    }

    reap_ready_jobs(active);
  }
}

}// namespace vkexec::detail
