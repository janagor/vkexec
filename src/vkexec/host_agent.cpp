#include "host_agent.hpp"

#include <vkexec/error.hpp>

#include <future>
#include <mutex>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace vkexec::detail {
namespace {

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  thread_local host_agent *tls_current_agent{ nullptr };

  struct current_agent_guard
  {
    host_agent *previous;

    explicit current_agent_guard(host_agent *agent) noexcept : previous(tls_current_agent)
    { tls_current_agent = agent; }

    current_agent_guard(current_agent_guard const &) = delete;
    auto operator=(current_agent_guard const &) -> current_agent_guard & = delete;
    current_agent_guard(current_agent_guard &&) = delete;
    auto operator=(current_agent_guard &&) -> current_agent_guard & = delete;

    ~current_agent_guard() { tls_current_agent = previous; }
  };

}// namespace

host_agent::host_agent()
{
  std::promise<void> ready;
  std::future<void> const started = ready.get_future();
  thread_ = std::jthread([this, signal = std::move(ready)](std::stop_token const & /*token*/) mutable -> void {
    thread_id_ = std::this_thread::get_id();
    signal.set_value();
    run();
  });
  started.wait();
}

host_agent::~host_agent() { shutdown(); }

auto host_agent::current() noexcept -> host_agent * { return tls_current_agent; }

auto host_agent::on_agent_thread() const noexcept -> bool { return std::this_thread::get_id() == thread_id_; }

auto host_agent::thread_id() const noexcept -> std::thread::id { return thread_id_; }

auto host_agent::enqueue(task_fn task) -> status
{
  if (!task) { return {}; }
  if (on_agent_thread()) {
    current_agent_guard const guard{ this };
    std::move(task)();
    return {};
  }

  {
    std::scoped_lock const lock(mutex_);
    if (shutting_down_) { return make_error(errc::invalid_argument, "host_agent enqueue after shutdown"); }
    pending_.push_back(std::move(task));
  }
  cv_.notify_one();
  return {};
}

auto host_agent::shutdown() -> void
{
  {
    std::scoped_lock const lock(mutex_);
    if (shutting_down_) { return; }
    shutting_down_ = true;
  }
  cv_.notify_all();
  thread_ = std::jthread{};
}

auto host_agent::run() -> void
{
  current_agent_guard const guard{ this };

  std::vector<task_fn> local;
  for (;;) {
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [this]() -> bool { return shutting_down_ || !pending_.empty(); });
      if (shutting_down_ && pending_.empty()) { return; }
      local.swap(pending_);
    }
    for (task_fn &task : local) {
      if (task) { std::move(task)(); }
    }
    local.clear();
  }
}

}// namespace vkexec::detail
