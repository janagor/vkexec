#ifndef VKEXEC_HOST_AGENT_HPP
#define VKEXEC_HOST_AGENT_HPP

#include <vkexec/detail/move_only_function.hpp>
#include <vkexec/result.hpp>
#include <vkexec/error.hpp>

#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace vkexec::detail {

/// Context-owned host run queue: `schedule` posts completions here.
class host_agent
{
public:
  using task_fn = move_only_function<void()>;

  host_agent();
  ~host_agent();

  host_agent(host_agent const &) = delete;
  auto operator=(host_agent const &) -> host_agent & = delete;
  host_agent(host_agent &&) = delete;
  auto operator=(host_agent &&) -> host_agent & = delete;

  /// Run `task` on the agent thread. Runs inline if already on that thread.
  auto enqueue(task_fn task) -> status;

  /// Drain remaining tasks and join the agent thread. Safe to call once.
  auto shutdown() -> void;

  [[nodiscard]] auto on_agent_thread() const noexcept -> bool;
  [[nodiscard]] auto thread_id() const noexcept -> std::thread::id;

  /// Agent currently executing a task on this thread, or null.
  [[nodiscard]] static auto current() noexcept -> host_agent *;

private:
  auto run() -> void;

  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<task_fn> pending_;
  bool shutting_down_{ false };
  std::thread::id thread_id_;
  std::jthread thread_;
};

}// namespace vkexec::detail

#endif// VKEXEC_HOST_AGENT_HPP
