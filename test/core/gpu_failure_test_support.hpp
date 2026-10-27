#ifndef VKEXEC_TEST_GPU_FAILURE_TEST_SUPPORT_HPP
#define VKEXEC_TEST_GPU_FAILURE_TEST_SUPPORT_HPP

#include "test_helpers.hpp"

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/object_synchronization.hpp>
#include <vkexec/detail/submission.hpp>
#include <vkexec/detail/synchronization.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/result.hpp>
#include <vkexec/submit_scope.hpp>

#include "vkexec/detail/retirement_wait.hpp"
#include "vkexec/detail/test_vulkan_dispatch.hpp"

#include <catch2/catch_test_macros.hpp>
#include <stdexec/execution.hpp>
#include <stdexec/stop_token.hpp>

#include <array>
#include <atomic>
#include <barrier>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <future>
#include <initializer_list>
#include <memory>
#include <optional>
#include <semaphore>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

inline namespace vkexec_gpu_failure_test_support {

constexpr auto k_completion_timeout = std::chrono::seconds(5);
constexpr int k_one_completion = 1;
constexpr int k_max_terminal_signals = 16;
constexpr int k_stress_threads = 4;
constexpr int k_expected_branch_submits = 2;
constexpr int k_default_stress_iterations = 8;
constexpr std::uint32_t k_default_stress_seed = 0x7A61C2D3U;
constexpr std::uint32_t k_random_multiplier = 1'664'525U;
constexpr std::uint32_t k_random_increment = 1'013'904'223U;
constexpr std::uint32_t k_cancellation_modes = 3U;
constexpr int k_fatal_exit_code = 73;

inline std::atomic<int> retirement_wait_calls{ 0 };
inline std::atomic<int> retirement_wait_failures{ 0 };

[[nodiscard]] inline auto VKAPI_CALL scripted_retirement_wait(VkDevice /*device*/,
  std::uint32_t /*count*/,
  VkFence const * /*fences*/,
  VkBool32 /*wait_all*/,
  std::uint64_t /*timeout*/) -> VkResult
{
  int const call = retirement_wait_calls.fetch_add(1, std::memory_order_relaxed);
  return call < retirement_wait_failures.load(std::memory_order_relaxed) ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
}

[[nodiscard]] inline auto VKAPI_CALL lost_retirement_wait(VkDevice /*device*/,
  std::uint32_t /*count*/,
  VkFence const * /*fences*/,
  VkBool32 /*wait_all*/,
  std::uint64_t /*timeout*/) -> VkResult
{ return VK_ERROR_DEVICE_LOST; }

[[nodiscard]] inline auto VKAPI_CALL scripted_queue_retirement_wait(VkQueue /*queue*/) -> VkResult
{
  int const call = retirement_wait_calls.fetch_add(1, std::memory_order_relaxed);
  return call < retirement_wait_failures.load(std::memory_order_relaxed) ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
}

[[nodiscard]] inline auto VKAPI_CALL lost_queue_retirement_wait(VkQueue /*queue*/) -> VkResult
{ return VK_ERROR_DEVICE_LOST; }

struct terminal_counts
{
  std::atomic<int> values{ 0 };
  std::atomic<int> errors{ 0 };
  std::atomic<int> stopped{ 0 };
  std::atomic<int> completions{ 0 };
  std::atomic<int> error_code{ 0 };
  std::counting_semaphore<k_max_terminal_signals> done{ 0 };

  auto signal(std::atomic<int> &counter) noexcept -> void
  {
    counter.fetch_add(1, std::memory_order_relaxed);
    if (completions.fetch_add(1, std::memory_order_acq_rel) != 0) { std::terminate(); }
    done.release();
  }

  [[nodiscard]] auto wait() -> bool { return done.try_acquire_for(k_completion_timeout); }
  [[nodiscard]] auto exactly_one() const noexcept -> bool
  { return completions.load(std::memory_order_acquire) == k_one_completion; }
};

struct recording_receiver
{
  using receiver_concept = stdexec::receiver_t;

  terminal_counts *counts{ nullptr };
  stdexec::inplace_stop_token token;

  auto set_value() && noexcept -> void { counts->signal(counts->values); }
  auto set_error(vkexec::error err) && noexcept -> void
  {
    counts->error_code.store(err.code.value(), std::memory_order_relaxed);
    counts->signal(counts->errors);
  }
  auto set_stopped() && noexcept -> void { counts->signal(counts->stopped); }

  [[nodiscard]] auto get_env() const noexcept -> decltype(stdexec::prop{ stdexec::get_stop_token, token })
  { return stdexec::prop{ stdexec::get_stop_token, token }; }
};

struct dispatch_reset
{
  explicit dispatch_reset(vkexec::detail::test_vulkan_dispatch &dispatch, vkexec::context const &ctx)
  {
    dispatch.device = ctx.device();
    dispatch.queues = { ctx.compute_queue(), ctx.graphics_queue(), ctx.present_queue() };
    vkexec::detail::set_test_vulkan_dispatch(&dispatch);
  }

  ~dispatch_reset() { vkexec::detail::set_test_vulkan_dispatch(nullptr); }

  dispatch_reset(dispatch_reset const &) = delete;
  auto operator=(dispatch_reset const &) -> dispatch_reset & = delete;
};

struct submit_gate
{
  std::binary_semaphore submitted{ 0 };
  std::binary_semaphore release{ 0 };
  std::atomic<int> calls{ 0 };
  PFN_vkQueueSubmit2 real_submit2{ nullptr };
};

inline std::atomic<submit_gate *> active_gate{ nullptr };
inline thread_local submit_gate *thread_gate = nullptr;
inline thread_local int *thread_submit_calls = nullptr;
inline std::atomic<PFN_vkQueueSubmit2> normal_submit2{ nullptr };

inline auto checkpoint_submit(VkQueue queue, std::uint32_t count, VkSubmitInfo const *submits, VkFence fence)
  -> VkResult
{
  VkResult const result = vkQueueSubmit(queue, count, submits, fence);
  if (thread_submit_calls != nullptr) { ++(*thread_submit_calls); }
  auto *gate = thread_gate != nullptr ? thread_gate : active_gate.load(std::memory_order_acquire);
  if (gate && result == VK_SUCCESS) {
    gate->calls.fetch_add(1, std::memory_order_relaxed);
    gate->submitted.release();
    gate->release.acquire();
  }
  return result;
}

inline auto checkpoint_submit2(VkQueue queue, std::uint32_t count, VkSubmitInfo2 const *submits, VkFence fence)
  -> VkResult
{
  auto *gate = thread_gate != nullptr ? thread_gate : active_gate.load(std::memory_order_acquire);
  auto *submit2 = gate != nullptr ? gate->real_submit2 : normal_submit2.load(std::memory_order_acquire);
  VkResult const result = submit2(queue, count, submits, fence);
  if (thread_submit_calls != nullptr) { ++(*thread_submit_calls); }
  if (gate != nullptr && result == VK_SUCCESS) {
    gate->calls.fetch_add(1, std::memory_order_relaxed);
    gate->submitted.release();
    gate->release.acquire();
  }
  return result;
}

struct gate_reset
{
  explicit gate_reset(submit_gate &gate) noexcept { active_gate.store(&gate, std::memory_order_release); }
  ~gate_reset() { active_gate.store(nullptr, std::memory_order_release); }
  gate_reset(gate_reset const &) = delete;
  auto operator=(gate_reset const &) -> gate_reset & = delete;
};

struct failing_record_step
{
  int *recorded{ nullptr };
  bool fail{ false };

  auto record(vkexec::context & /*ctx*/, VkCommandBuffer /*cmd*/, vkexec::detail::pass_cleanup & /*cleanup*/) const
    -> vkexec::status
  {
    ++(*recorded);
    if (fail) { return vkexec::fail(vkexec::errc::invalid_argument); }
    return {};
  }
};

[[nodiscard]] inline auto recorded_scope(vkexec::context &ctx) -> vkexec::detail::submit_scope
{
  auto opened = vkexec::detail::submit_scope::open(ctx);
  REQUIRE(opened.has_value());
  auto scope = vkexec::expected_take(opened);
  REQUIRE(scope.end_recording().has_value());
  return scope;
}

[[nodiscard]] inline auto submit2_proc(vkexec::context const &ctx) -> PFN_vkQueueSubmit2
{
  auto const *name = ctx.capabilities().api_version >= VK_API_VERSION_1_3 ? "vkQueueSubmit2" : "vkQueueSubmit2KHR";
  return reinterpret_cast<PFN_vkQueueSubmit2>(vkGetDeviceProcAddr(ctx.device(), name));
}

inline std::atomic<VkResult> injected_submit_result{ VK_ERROR_DEVICE_LOST };
inline std::atomic<int> branch_submit_calls{ 0 };
inline std::atomic<int> branch_accepted_submits{ 0 };
inline std::atomic<int> branch_rejected_submits{ 0 };
inline PFN_vkQueueSubmit2 branch_real_submit2{ nullptr };

[[nodiscard]] inline auto
  fail_second_submit(VkQueue queue, std::uint32_t count, VkSubmitInfo const *submits, VkFence fence) -> VkResult
{
  if (branch_submit_calls.fetch_add(1, std::memory_order_relaxed) == 0) {
    VkResult const result = vkQueueSubmit(queue, count, submits, fence);
    if (result == VK_SUCCESS) { branch_accepted_submits.fetch_add(1, std::memory_order_relaxed); }
    return result;
  }
  branch_rejected_submits.fetch_add(1, std::memory_order_relaxed);
  return VK_ERROR_DEVICE_LOST;
}

[[nodiscard]] inline auto
  fail_second_submit2(VkQueue queue, std::uint32_t count, VkSubmitInfo2 const *submits, VkFence fence) -> VkResult
{
  if (branch_submit_calls.fetch_add(1, std::memory_order_relaxed) == 0) {
    VkResult const result = branch_real_submit2(queue, count, submits, fence);
    if (result == VK_SUCCESS) { branch_accepted_submits.fetch_add(1, std::memory_order_relaxed); }
    return result;
  }
  branch_rejected_submits.fetch_add(1, std::memory_order_relaxed);
  return VK_ERROR_DEVICE_LOST;
}

[[nodiscard]] inline auto
  failing_submit(VkQueue /*queue*/, std::uint32_t /*count*/, VkSubmitInfo const * /*submits*/, VkFence /*fence*/)
    -> VkResult
{ return injected_submit_result.load(std::memory_order_relaxed); }

[[nodiscard]] inline auto
  failing_submit2(VkQueue /*queue*/, std::uint32_t /*count*/, VkSubmitInfo2 const * /*submits*/, VkFence /*fence*/)
    -> VkResult
{ return injected_submit_result.load(std::memory_order_relaxed); }

inline std::atomic<int> wait_failures{ 0 };
inline std::atomic<VkResult> injected_wait_result{ VK_ERROR_DEVICE_LOST };
inline std::atomic<std::binary_semaphore *> wait_failure_seen{ nullptr };

[[nodiscard]] inline auto
  fail_wait_once(VkDevice device, std::uint32_t count, VkFence const *fences, VkBool32 wait_all, std::uint64_t timeout)
    -> VkResult
{
  if (wait_failures.fetch_add(1, std::memory_order_relaxed) == 0) {
    if (auto *checkpoint = wait_failure_seen.load(std::memory_order_acquire)) { checkpoint->release(); }
    return injected_wait_result.load(std::memory_order_relaxed);
  }
  return vkWaitForFences(device, count, fences, wait_all, timeout);
}

[[nodiscard]] inline auto fail_fence_status(VkDevice /*device*/, VkFence /*fence*/) -> VkResult
{ return VK_ERROR_DEVICE_LOST; }

[[nodiscard]] inline auto fail_queue_wait_idle(VkQueue /*queue*/) -> VkResult { return VK_ERROR_DEVICE_LOST; }

#if VKEXEC_HAS_EXCEPTIONS
[[nodiscard]] inline auto fail_enqueue() -> bool { return true; }
#endif

}// namespace vkexec_gpu_failure_test_support


#endif// VKEXEC_TEST_GPU_FAILURE_TEST_SUPPORT_HPP
