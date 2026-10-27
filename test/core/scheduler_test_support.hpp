#ifndef VKEXEC_TEST_SCHEDULER_TEST_SUPPORT_HPP
#define VKEXEC_TEST_SCHEDULER_TEST_SUPPORT_HPP

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/bind_resources.hpp>
#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/object_synchronization.hpp>
#include <vkexec/detail/sender_expr.hpp>
#include <vkexec/domain.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/result.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit_scope.hpp>

#include <stdexec/execution.hpp>
#include <stdexec/stop_token.hpp>
#include <vulkan/vulkan_core.h>

#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#if VKEXEC_HAS_EXCEPTIONS
#include <exception>
#include <stdexcept>
#endif
#include <future>
#include <memory>
#include <optional>
#include <semaphore>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

inline namespace vkexec_scheduler_test_support {

struct empty_receiver_env
{
};

inline auto signal_promise(std::promise<void> &promise) noexcept -> void
{
#if VKEXEC_HAS_EXCEPTIONS
  try {
    promise.set_value();
  } catch (std::future_error const &err) {
    (void)err;// already satisfied
  }
#else
  promise.set_value();
#endif
}

struct completion_probe_receiver
{
  using receiver_concept = stdexec::receiver_t;

  std::promise<void> *completed{ nullptr };

  auto signal_done() const noexcept -> void { signal_promise(*completed); }

  auto set_value() const && noexcept -> void { signal_done(); }

  auto set_error(vkexec::error const & /*err*/) const && noexcept -> void { signal_done(); }

#if VKEXEC_HAS_EXCEPTIONS
  auto set_error(std::exception_ptr const & /*exception*/) const && noexcept -> void { signal_done(); }
#endif

  auto set_stopped() const && noexcept -> void { signal_done(); }

  // cppcheck-suppress functionStatic
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  [[nodiscard]] auto get_env() const noexcept -> empty_receiver_env { return {}; }
};

struct schedule_outcome_receiver
{
  using receiver_concept = stdexec::receiver_t;

  std::atomic<int> *outcome{ nullptr };
  std::binary_semaphore *completed{ nullptr };

  auto complete(int value) const noexcept -> void
  {
    // NOLINTNEXTLINE(misc-const-correctness)
    int expected = 0;
    if (!outcome->compare_exchange_strong(expected, value)) { std::terminate(); }
    completed->release();
  }

  auto set_value() const && noexcept -> void { complete(1); }
  auto set_error(vkexec::error const & /*err*/) const && noexcept -> void { complete(2); }
  auto set_stopped() const && noexcept -> void { complete(3); }

  // cppcheck-suppress functionStatic
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  [[nodiscard]] auto get_env() const noexcept -> empty_receiver_env { return {}; }
};

struct move_only_pass_step
{
  std::unique_ptr<int> state;

  explicit move_only_pass_step(int initial_value) : state(std::make_unique<int>(initial_value)) {}
  ~move_only_pass_step() = default;
  move_only_pass_step(move_only_pass_step &&) noexcept = default;
  auto operator=(move_only_pass_step &&) noexcept -> move_only_pass_step & = default;
  move_only_pass_step(move_only_pass_step const &) = delete;
  auto operator=(move_only_pass_step const &) -> move_only_pass_step & = delete;

  static auto record(vkexec::context & /*ctx*/, VkCommandBuffer /*cmd*/, vkexec::detail::pass_cleanup & /*cleanup*/)
    -> vkexec::status
  { return {}; }
};

struct counting_pass_step
{
  int *record_count{ nullptr };
  bool should_fail{ false };

  auto record(vkexec::context & /*ctx*/, VkCommandBuffer /*cmd*/, vkexec::detail::pass_cleanup & /*cleanup*/) const
    -> vkexec::status
  {
    ++(*record_count);
    if (should_fail) { return vkexec::fail(vkexec::errc::invalid_argument); }
    return {};
  }
};

struct informed_pass_step
{
  decltype(vkexec::uses()) resources{};
  vkexec::pass_recording_info *observed{};
  bool should_fail{};

  auto record(vkexec::context & /*ctx*/,
    VkCommandBuffer /*cmd*/,
    vkexec::detail::pass_cleanup & /*cleanup*/,
    vkexec::pass_recording_info const &info) const -> vkexec::status
  {
    (void)info.ensure_submission_id();
    *observed = info;
    if (should_fail) { return vkexec::fail(vkexec::errc::invalid_argument); }
    return {};
  }
};

struct completion_commit_step
{
  int *commits{};
  bool should_fail{};

  auto record(vkexec::context & /*ctx*/, VkCommandBuffer /*cmd*/, vkexec::detail::pass_cleanup & /*cleanup*/) const
    -> vkexec::status
  {
    if (should_fail) { return vkexec::fail(vkexec::errc::invalid_argument); }
    return {};
  }

  auto commit_after_gpu() const -> void { ++*commits; }
};

static_assert(vkexec::detail::static_pass_step<informed_pass_step>);

struct dual_record_pass_step
{
  int *selected{};

  auto record(vkexec::context & /*ctx*/, VkCommandBuffer /*cmd*/, vkexec::detail::pass_cleanup & /*cleanup*/) const
    -> vkexec::status
  {
    *selected = k_legacy_signature;
    return {};
  }

  auto record(vkexec::context & /*ctx*/,
    VkCommandBuffer /*cmd*/,
    vkexec::detail::pass_cleanup & /*cleanup*/,
    vkexec::pass_recording_info const & /*info*/) const -> vkexec::status
  {
    *selected = k_context_signature;
    return {};
  }

  static constexpr int k_legacy_signature = 3;
  static constexpr int k_context_signature = 4;
};

template<class AfterGpu> [[nodiscard]] auto make_empty_pass_step(AfterGpu after_gpu)
{
  return vkexec::detail::make_callback_pass_step(
    [](vkexec::context & /*host*/,
      VkCommandBuffer /*cmd*/,
      vkexec::detail::pass_cleanup & /*cleanup*/) -> vkexec::status { return {}; },
    std::move(after_gpu));
}

struct release_flag
{
  std::atomic_bool *gate{ nullptr };

  explicit release_flag(std::atomic_bool &value) noexcept : gate(&value) {}

  release_flag(release_flag const &) = delete;
  auto operator=(release_flag const &) -> release_flag & = delete;
  release_flag(release_flag &&) = delete;
  auto operator=(release_flag &&) -> release_flag & = delete;

  auto release() noexcept -> void
  {
    if (gate == nullptr) { return; }
    gate->store(true, std::memory_order_release);
    gate->notify_all();
    gate = nullptr;
  }

  ~release_flag() noexcept { release(); }
};

}// namespace vkexec_scheduler_test_support

template<class Sender, class CompletionTag>
concept advertises_completion_scheduler =
  requires(Sender const &sndr) { stdexec::get_completion_scheduler<CompletionTag>(stdexec::get_env(sndr)); };

using move_only_graph_t = vkexec::pass_graph_sender<move_only_pass_step>;

template<class Graph>
concept rvalue_connectable_graph = requires(Graph graph, completion_probe_receiver receiver) {
  stdexec::connect(std::move(graph), std::move(receiver));
};

template<class Graph>
concept const_lvalue_connectable_graph =
  requires(Graph const &graph, completion_probe_receiver receiver) { stdexec::connect(graph, std::move(receiver)); };

template<class Sender, class Closure>
concept pipeable_with =
  requires(Sender &&sender, Closure &&closure) { std::forward<Sender>(sender) | std::forward<Closure>(closure); };

using move_only_closure_t = decltype(vkexec::make_pass_adaptor(std::declval<move_only_pass_step>()));

static_assert(vkexec::detail::static_pass_step<move_only_pass_step>);
static_assert(std::move_constructible<move_only_graph_t>);
static_assert(!std::copy_constructible<move_only_graph_t>);
static_assert(rvalue_connectable_graph<move_only_graph_t>);
static_assert(!const_lvalue_connectable_graph<move_only_graph_t>);
static_assert(pipeable_with<vkexec::schedule_sender, move_only_closure_t>);
static_assert(!pipeable_with<vkexec::schedule_sender, move_only_closure_t &>);
static_assert(advertises_completion_scheduler<vkexec::schedule_sender, stdexec::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::schedule_sender, stdexec::set_error_t>);
static_assert(!advertises_completion_scheduler<vkexec::schedule_sender, stdexec::set_stopped_t>);
static_assert(std::same_as<vkexec::schedule_sender::completion_signatures,
  stdexec::
    completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(vkexec::error), stdexec::set_stopped_t()>>);
static_assert(std::same_as<vkexec::detail::submit_and_wait_sender::completion_signatures,
  stdexec::
    completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(vkexec::error), stdexec::set_stopped_t()>>);

static_assert(advertises_completion_scheduler<vkexec::pass_graph_sender<>, stdexec::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::pass_graph_sender<>, stdexec::set_error_t>);

static_assert(!advertises_completion_scheduler<vkexec::detail::enter_submit_scope_sender, stdexec::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::detail::submit_and_wait_sender, stdexec::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::detail::submit_fence_sender, stdexec::set_value_t>);

static_assert(vkexec::vkexec_predecessor<vkexec::schedule_sender>);
static_assert(vkexec::vkexec_predecessor<vkexec::pass_graph_sender<>>);
static_assert(std::move_constructible<vkexec::dynamic_pass_graph_sender>);
static_assert(!std::copy_constructible<vkexec::dynamic_pass_graph_sender>);
static_assert(vkexec::detail::is_pass_graph_sender_v<vkexec::dynamic_pass_graph_sender>);
static_assert(vkexec::vkexec_predecessor<vkexec::dynamic_pass_graph_sender>);

#endif// VKEXEC_TEST_SCHEDULER_TEST_SUPPORT_HPP
