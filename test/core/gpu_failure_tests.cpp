// NOLINTBEGIN(readability-function-cognitive-complexity)

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

namespace {

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

std::atomic<int> retirement_wait_calls{ 0 };
std::atomic<int> retirement_wait_failures{ 0 };

[[nodiscard]] auto VKAPI_CALL scripted_retirement_wait(VkDevice /*device*/,
  std::uint32_t /*count*/,
  VkFence const * /*fences*/,
  VkBool32 /*wait_all*/,
  std::uint64_t /*timeout*/) -> VkResult
{
  int const call = retirement_wait_calls.fetch_add(1, std::memory_order_relaxed);
  return call < retirement_wait_failures.load(std::memory_order_relaxed) ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
}

[[nodiscard]] auto VKAPI_CALL lost_retirement_wait(VkDevice /*device*/,
  std::uint32_t /*count*/,
  VkFence const * /*fences*/,
  VkBool32 /*wait_all*/,
  std::uint64_t /*timeout*/) -> VkResult
{ return VK_ERROR_DEVICE_LOST; }

[[nodiscard]] auto VKAPI_CALL scripted_queue_retirement_wait(VkQueue /*queue*/) -> VkResult
{
  int const call = retirement_wait_calls.fetch_add(1, std::memory_order_relaxed);
  return call < retirement_wait_failures.load(std::memory_order_relaxed) ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
}

[[nodiscard]] auto VKAPI_CALL lost_queue_retirement_wait(VkQueue /*queue*/) -> VkResult { return VK_ERROR_DEVICE_LOST; }

TEST_CASE("retirement wait retries ordinary errors and preserves the first error", "[vkexec][gpu][failure]")
{
  constexpr int k_first_wait_fails = 1;
  constexpr int k_first_two_waits_fail = 2;
  for (int const failures : { k_first_wait_fails, k_first_two_waits_fail }) {
    retirement_wait_calls.store(0, std::memory_order_relaxed);
    retirement_wait_failures.store(failures, std::memory_order_relaxed);
    REQUIRE(vkexec::detail::wait_for_retirement(VK_NULL_HANDLE, 0, nullptr, scripted_retirement_wait)
            == VK_ERROR_OUT_OF_HOST_MEMORY);
    REQUIRE(retirement_wait_calls.load() == failures + 1);
  }
}

TEST_CASE("retirement wait accepts device loss", "[vkexec][gpu][failure]")
{
  REQUIRE(
    vkexec::detail::wait_for_retirement(VK_NULL_HANDLE, 0, nullptr, lost_retirement_wait) == VK_ERROR_DEVICE_LOST);
}

TEST_CASE("queue retirement retries errors and accepts device loss", "[vkexec][gpu][failure]")
{
  constexpr int k_first_two_waits_fail = 2;
  retirement_wait_calls.store(0, std::memory_order_relaxed);
  retirement_wait_failures.store(k_first_two_waits_fail, std::memory_order_relaxed);
  REQUIRE(vkexec::detail::wait_for_queue_retirement(VK_NULL_HANDLE, scripted_queue_retirement_wait)
          == VK_ERROR_OUT_OF_HOST_MEMORY);
  REQUIRE(retirement_wait_calls.load() == k_first_two_waits_fail + 1);
  REQUIRE(
    vkexec::detail::wait_for_queue_retirement(VK_NULL_HANDLE, lost_queue_retirement_wait) == VK_ERROR_DEVICE_LOST);
}

#if !defined(_WIN32)
TEST_CASE("persistent retirement wait errors invoke the fatal policy", "[vkexec][gpu][failure]")
{
  pid_t const child = fork();
  REQUIRE(child >= 0);
  if (child == 0) {
    execl(VKEXEC_RETIREMENT_FATAL_HELPER_PATH, VKEXEC_RETIREMENT_FATAL_HELPER_PATH, nullptr);
    std::_Exit(EXIT_FAILURE);
  }
  int status = 0;
  REQUIRE(waitpid(child, &status, 0) == child);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == k_fatal_exit_code);
}
#endif

template<class Value> [[nodiscard]] auto environment_number(char const *name, Value fallback) -> Value
{
#ifdef _MSC_VER
  char *input = nullptr;
  std::size_t input_size = 0;
  if (_dupenv_s(&input, &input_size, name) != 0) { return fallback; }
  std::unique_ptr<char, decltype(&std::free)> owned_input{ input, &std::free };
  if (!owned_input) { return fallback; }
  std::string_view const value{ owned_input.get() };
#else
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  char const *const input = std::getenv(name);
  if (input == nullptr) { return fallback; }
  std::string_view const value{ input };
#endif
  Value parsed{};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  auto const result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  return result.ec == std::errc{} && result.ptr == value.data() + value.size() && parsed > 0 ? parsed : fallback;
}

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

std::atomic<submit_gate *> active_gate{ nullptr };
thread_local submit_gate *thread_gate = nullptr;
thread_local int *thread_submit_calls = nullptr;
std::atomic<PFN_vkQueueSubmit2> normal_submit2{ nullptr };

auto checkpoint_submit(VkQueue queue, std::uint32_t count, VkSubmitInfo const *submits, VkFence fence) -> VkResult
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

auto checkpoint_submit2(VkQueue queue, std::uint32_t count, VkSubmitInfo2 const *submits, VkFence fence) -> VkResult
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

[[nodiscard]] auto recorded_scope(vkexec::context &ctx) -> vkexec::detail::submit_scope
{
  auto opened = vkexec::detail::submit_scope::open(ctx);
  REQUIRE(opened.has_value());
  auto scope = vkexec::expected_take(opened);
  REQUIRE(scope.end_recording().has_value());
  return scope;
}

[[nodiscard]] auto submit2_proc(vkexec::context const &ctx) -> PFN_vkQueueSubmit2
{
  auto const *name = ctx.capabilities().api_version >= VK_API_VERSION_1_3 ? "vkQueueSubmit2" : "vkQueueSubmit2KHR";
  return reinterpret_cast<PFN_vkQueueSubmit2>(vkGetDeviceProcAddr(ctx.device(), name));
}

std::atomic<VkResult> injected_submit_result{ VK_ERROR_DEVICE_LOST };
std::atomic<int> branch_submit_calls{ 0 };
std::atomic<int> branch_accepted_submits{ 0 };
std::atomic<int> branch_rejected_submits{ 0 };
PFN_vkQueueSubmit2 branch_real_submit2{ nullptr };

[[nodiscard]] auto fail_second_submit(VkQueue queue, std::uint32_t count, VkSubmitInfo const *submits, VkFence fence)
  -> VkResult
{
  if (branch_submit_calls.fetch_add(1, std::memory_order_relaxed) == 0) {
    VkResult const result = vkQueueSubmit(queue, count, submits, fence);
    if (result == VK_SUCCESS) { branch_accepted_submits.fetch_add(1, std::memory_order_relaxed); }
    return result;
  }
  branch_rejected_submits.fetch_add(1, std::memory_order_relaxed);
  return VK_ERROR_DEVICE_LOST;
}

[[nodiscard]] auto fail_second_submit2(VkQueue queue, std::uint32_t count, VkSubmitInfo2 const *submits, VkFence fence)
  -> VkResult
{
  if (branch_submit_calls.fetch_add(1, std::memory_order_relaxed) == 0) {
    VkResult const result = branch_real_submit2(queue, count, submits, fence);
    if (result == VK_SUCCESS) { branch_accepted_submits.fetch_add(1, std::memory_order_relaxed); }
    return result;
  }
  branch_rejected_submits.fetch_add(1, std::memory_order_relaxed);
  return VK_ERROR_DEVICE_LOST;
}

[[nodiscard]] auto
  failing_submit(VkQueue /*queue*/, std::uint32_t /*count*/, VkSubmitInfo const * /*submits*/, VkFence /*fence*/)
    -> VkResult
{ return injected_submit_result.load(std::memory_order_relaxed); }

[[nodiscard]] auto
  failing_submit2(VkQueue /*queue*/, std::uint32_t /*count*/, VkSubmitInfo2 const * /*submits*/, VkFence /*fence*/)
    -> VkResult
{ return injected_submit_result.load(std::memory_order_relaxed); }

std::atomic<int> wait_failures{ 0 };
std::atomic<VkResult> injected_wait_result{ VK_ERROR_DEVICE_LOST };
std::atomic<std::binary_semaphore *> wait_failure_seen{ nullptr };

[[nodiscard]] auto
  fail_wait_once(VkDevice device, std::uint32_t count, VkFence const *fences, VkBool32 wait_all, std::uint64_t timeout)
    -> VkResult
{
  if (wait_failures.fetch_add(1, std::memory_order_relaxed) == 0) {
    if (auto *checkpoint = wait_failure_seen.load(std::memory_order_acquire)) { checkpoint->release(); }
    return injected_wait_result.load(std::memory_order_relaxed);
  }
  return vkWaitForFences(device, count, fences, wait_all, timeout);
}

[[nodiscard]] auto fail_fence_status(VkDevice /*device*/, VkFence /*fence*/) -> VkResult
{ return VK_ERROR_DEVICE_LOST; }

[[nodiscard]] auto fail_queue_wait_idle(VkQueue /*queue*/) -> VkResult { return VK_ERROR_DEVICE_LOST; }

#if VKEXEC_HAS_EXCEPTIONS
[[nodiscard]] auto fail_enqueue() -> bool { return true; }
#endif

}// namespace

TEST_CASE("submit_fence observes pre-requested cancellation before Vulkan submission", "[vkexec][gpu][failure]")
{
  auto ctx = vkexec::test::require_context();
  submit_gate gate;
  gate.real_submit2 = submit2_proc(*ctx);
  vkexec::detail::test_vulkan_dispatch dispatch;
  dispatch.queue_submit = checkpoint_submit;
  dispatch.queue_submit2 = gate.real_submit2 ? checkpoint_submit2 : nullptr;
  gate_reset const active{ gate };
  dispatch_reset const installed{ dispatch, *ctx };

  stdexec::inplace_stop_source source;
  source.request_stop();
  terminal_counts counts;
  auto operation = stdexec::connect(vkexec::detail::submit_fence(recorded_scope(*ctx)),
    recording_receiver{ .counts = &counts, .token = source.get_token() });
  stdexec::start(operation);
  REQUIRE(counts.wait());
  REQUIRE(counts.stopped.load() == k_one_completion);
  REQUIRE(counts.exactly_one());
  REQUIRE(gate.calls.load() == 0);
}

TEST_CASE("submit_fence observes cancellation after successful queue submission", "[vkexec][gpu][failure]")
{
  auto ctx = vkexec::test::require_context();
  submit_gate gate;
  gate.real_submit2 = submit2_proc(*ctx);
  vkexec::detail::test_vulkan_dispatch dispatch;
  dispatch.queue_submit = checkpoint_submit;
  dispatch.queue_submit2 = gate.real_submit2 ? checkpoint_submit2 : nullptr;
  gate_reset const active{ gate };
  dispatch_reset const installed{ dispatch, *ctx };

  stdexec::inplace_stop_source source;
  terminal_counts counts;
  auto operation = stdexec::connect(vkexec::detail::submit_fence(recorded_scope(*ctx)),
    recording_receiver{ .counts = &counts, .token = source.get_token() });
  std::thread starter{ [&operation]() -> void { stdexec::start(operation); } };
  bool const submitted = gate.submitted.try_acquire_for(k_completion_timeout);
  source.request_stop();
  gate.release.release();
  starter.join();

  REQUIRE(submitted);
  REQUIRE(counts.wait());
  REQUIRE(gate.calls.load() == k_one_completion);
  REQUIRE(counts.stopped.load() == k_one_completion);
  REQUIRE(counts.exactly_one());
}

TEST_CASE("submit_and_wait ignores cancellation after queue submission", "[vkexec][gpu][failure]")
{
  auto ctx = vkexec::test::require_context();
  submit_gate gate;
  gate.real_submit2 = submit2_proc(*ctx);
  vkexec::detail::test_vulkan_dispatch dispatch;
  dispatch.queue_submit = checkpoint_submit;
  dispatch.queue_submit2 = gate.real_submit2 ? checkpoint_submit2 : nullptr;
  gate_reset const active{ gate };
  dispatch_reset const installed{ dispatch, *ctx };

  stdexec::inplace_stop_source source;
  terminal_counts counts;
  auto operation = stdexec::connect(vkexec::detail::submit_and_wait(recorded_scope(*ctx)),
    recording_receiver{ .counts = &counts, .token = source.get_token() });
  std::thread starter{ [&operation]() -> void { stdexec::start(operation); } };
  bool const submitted = gate.submitted.try_acquire_for(k_completion_timeout);
  source.request_stop();
  gate.release.release();
  starter.join();

  REQUIRE(submitted);
  REQUIRE(counts.wait());
  REQUIRE(counts.values.load() == k_one_completion);
  REQUIRE(counts.exactly_one());
}

TEST_CASE("submit_fence ignores cancellation after terminal notification", "[vkexec][gpu][failure]")
{
  auto ctx = vkexec::test::require_context();
  stdexec::inplace_stop_source source;
  terminal_counts counts;
  auto operation = stdexec::connect(vkexec::detail::submit_fence(recorded_scope(*ctx)),
    recording_receiver{ .counts = &counts, .token = source.get_token() });
  stdexec::start(operation);
  REQUIRE(counts.wait());
  source.request_stop();
  ctx.reset();
  REQUIRE(counts.values.load() == k_one_completion);
  REQUIRE(counts.exactly_one());
}

TEST_CASE("submit_fence holds resources until a timeline-blocked queue retires", "[vkexec][gpu][failure]")
{
  constexpr std::uint64_t k_release_value = 1;
  auto ctx = vkexec::test::require_context();
  if (!ctx->capabilities().timeline_semaphore) { SKIP("Timeline semaphores are unavailable"); }
  auto signal_semaphore =
    reinterpret_cast<PFN_vkSignalSemaphore>(vkGetDeviceProcAddr(ctx->device(), "vkSignalSemaphore"));
  if (signal_semaphore == nullptr) {
    signal_semaphore =
      reinterpret_cast<PFN_vkSignalSemaphore>(vkGetDeviceProcAddr(ctx->device(), "vkSignalSemaphoreKHR"));
  }
  if (signal_semaphore == nullptr) { SKIP("Timeline semaphore host signaling is unavailable"); }

  VkSemaphoreTypeCreateInfo type_info{};
  type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphore_info.pNext = &type_info;
  VkSemaphore gate{ VK_NULL_HANDLE };
  REQUIRE(vkCreateSemaphore(ctx->device(), &semaphore_info, nullptr, &gate) == VK_SUCCESS);

  auto allocated = ctx->allocate_command_buffer();
  REQUIRE(allocated.has_value());
  VkCommandBuffer blocker = vkexec::expected_take(allocated);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  REQUIRE(vkBeginCommandBuffer(blocker, &begin) == VK_SUCCESS);
  REQUIRE(vkEndCommandBuffer(blocker) == VK_SUCCESS);
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence blocker_fence{ VK_NULL_HANDLE };
  REQUIRE(vkCreateFence(ctx->device(), &fence_info, nullptr, &blocker_fence) == VK_SUCCESS);

  std::array<VkCommandBuffer, 1> const commands{ blocker };
  std::array<vkexec::semaphore_submit, 1> const waits{
    vkexec::semaphore_submit{ .semaphore = gate, .value = k_release_value },
  };
  REQUIRE(ctx->submit(vkexec::queue_submit{ .command_buffers = commands, .waits = waits, .fence = blocker_fence })
      .has_value());

  stdexec::inplace_stop_source source;
  terminal_counts counts;
  auto operation = stdexec::connect(vkexec::detail::submit_fence(recorded_scope(*ctx)),
    recording_receiver{ .counts = &counts, .token = source.get_token() });
  stdexec::start(operation);
  source.request_stop();
  VkResult const blocker_status = vkGetFenceStatus(ctx->device(), blocker_fence);
  bool const completed_while_blocked = counts.completions.load(std::memory_order_acquire) != 0;

  VkSemaphoreSignalInfo signal_info{};
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.semaphore = gate;
  signal_info.value = k_release_value;
  VkResult const signaled = signal_semaphore(ctx->device(), &signal_info);
  bool const completed = counts.wait();
  VkResult const blocker_wait = vkWaitForFences(ctx->device(), 1, &blocker_fence, VK_TRUE, UINT64_MAX);
  ctx->free_command_buffer(blocker);
  vkDestroyFence(ctx->device(), blocker_fence, nullptr);
  vkDestroySemaphore(ctx->device(), gate, nullptr);

  REQUIRE(signaled == VK_SUCCESS);
  REQUIRE(blocker_status == VK_NOT_READY);
  REQUIRE(blocker_wait == VK_SUCCESS);
  REQUIRE_FALSE(completed_while_blocked);
  REQUIRE(completed);
  REQUIRE(counts.stopped.load() == k_one_completion);
  REQUIRE(counts.exactly_one());
}

TEST_CASE("injected wait failure keeps timeline-blocked GPU work alive until retirement", "[vkexec][gpu][failure]")
{
  constexpr std::uint64_t k_release_value = 1;
  auto ctx = vkexec::test::require_context();
  if (!ctx->capabilities().timeline_semaphore) { SKIP("Timeline semaphores are unavailable"); }
  auto signal_semaphore =
    reinterpret_cast<PFN_vkSignalSemaphore>(vkGetDeviceProcAddr(ctx->device(), "vkSignalSemaphore"));
  if (signal_semaphore == nullptr) {
    signal_semaphore =
      reinterpret_cast<PFN_vkSignalSemaphore>(vkGetDeviceProcAddr(ctx->device(), "vkSignalSemaphoreKHR"));
  }
  if (signal_semaphore == nullptr) { SKIP("Timeline semaphore host signaling is unavailable"); }

  auto blocker = recorded_scope(*ctx);
  VkSemaphoreTypeCreateInfo type_info{};
  type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphore_info.pNext = &type_info;
  VkSemaphore gate{ VK_NULL_HANDLE };
  REQUIRE(vkCreateSemaphore(ctx->device(), &semaphore_info, nullptr, &gate) == VK_SUCCESS);
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence blocker_fence{ VK_NULL_HANDLE };
  REQUIRE(vkCreateFence(ctx->device(), &fence_info, nullptr, &blocker_fence) == VK_SUCCESS);
  std::array<VkCommandBuffer, 1> const commands{ blocker.cmd };
  std::array<vkexec::semaphore_submit, 1> const waits{
    vkexec::semaphore_submit{ .semaphore = gate, .value = k_release_value },
  };
  REQUIRE(ctx->submit(vkexec::queue_submit{ .command_buffers = commands, .waits = waits, .fence = blocker_fence })
      .has_value());

  wait_failures.store(0, std::memory_order_relaxed);
  injected_wait_result.store(VK_ERROR_DEVICE_LOST, std::memory_order_relaxed);
  std::binary_semaphore failure_seen{ 0 };
  wait_failure_seen.store(&failure_seen, std::memory_order_release);
  vkexec::detail::test_vulkan_dispatch dispatch;
  dispatch.wait_for_fences = fail_wait_once;
  dispatch_reset const installed{ dispatch, *ctx };
  stdexec::inplace_stop_source source;
  terminal_counts counts;
  auto operation = stdexec::connect(vkexec::detail::submit_fence(recorded_scope(*ctx)),
    recording_receiver{ .counts = &counts, .token = source.get_token() });
  stdexec::start(operation);
  bool const injected = failure_seen.try_acquire_for(k_completion_timeout);
  source.request_stop();
  VkResult const blocker_status = vkGetFenceStatus(ctx->device(), blocker_fence);
  bool const completed_while_blocked = counts.completions.load(std::memory_order_acquire) != 0;

  VkSemaphoreSignalInfo signal_info{};
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.semaphore = gate;
  signal_info.value = k_release_value;
  VkResult const signaled = signal_semaphore(ctx->device(), &signal_info);
  bool const completed = counts.wait();
  VkResult const blocker_wait = vkWaitForFences(ctx->device(), 1, &blocker_fence, VK_TRUE, UINT64_MAX);
  wait_failure_seen.store(nullptr, std::memory_order_release);
  blocker.release();
  vkDestroyFence(ctx->device(), blocker_fence, nullptr);
  vkDestroySemaphore(ctx->device(), gate, nullptr);

  REQUIRE(injected);
  REQUIRE(blocker_status == VK_NOT_READY);
  REQUIRE_FALSE(completed_while_blocked);
  REQUIRE(signaled == VK_SUCCESS);
  REQUIRE(blocker_wait == VK_SUCCESS);
  REQUIRE(completed);
  REQUIRE(counts.errors.load() == k_one_completion);
  REQUIRE(counts.error_code.load() == VK_ERROR_DEVICE_LOST);
  REQUIRE(counts.exactly_one());
  auto retry = vkexec::test::sync_wait_sender(vkexec::detail::submit_and_wait(recorded_scope(*ctx)));
  REQUIRE(vkexec::test::sync_wait_completed(retry));
}

TEST_CASE("pending GPU submissions outlive the public context and retire once", "[vkexec][gpu][failure]")
{
  constexpr std::uint64_t k_release_value = 1;
  auto ctx = vkexec::test::require_context();
  if (!ctx->capabilities().timeline_semaphore) { SKIP("Timeline semaphores are unavailable"); }
  auto signal_semaphore =
    reinterpret_cast<PFN_vkSignalSemaphore>(vkGetDeviceProcAddr(ctx->device(), "vkSignalSemaphore"));
  if (signal_semaphore == nullptr) {
    signal_semaphore =
      reinterpret_cast<PFN_vkSignalSemaphore>(vkGetDeviceProcAddr(ctx->device(), "vkSignalSemaphoreKHR"));
  }
  if (signal_semaphore == nullptr) { SKIP("Timeline semaphore host signaling is unavailable"); }

  auto retained = vkexec::detail::context_access::state(*ctx);
  auto retired = vkexec::detail::retirement_future(retained);
  VkDevice const device = ctx->device();
  auto blocker = recorded_scope(*ctx);
  VkSemaphoreTypeCreateInfo type_info{};
  type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphore_info.pNext = &type_info;
  VkSemaphore gate{ VK_NULL_HANDLE };
  REQUIRE(vkCreateSemaphore(device, &semaphore_info, nullptr, &gate) == VK_SUCCESS);
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence blocker_fence{ VK_NULL_HANDLE };
  REQUIRE(vkCreateFence(device, &fence_info, nullptr, &blocker_fence) == VK_SUCCESS);
  std::array<VkCommandBuffer, 1> const commands{ blocker.cmd };
  std::array<vkexec::semaphore_submit, 1> const waits{
    vkexec::semaphore_submit{ .semaphore = gate, .value = k_release_value },
  };
  REQUIRE(ctx->submit(vkexec::queue_submit{ .command_buffers = commands, .waits = waits, .fence = blocker_fence })
      .has_value());

  {
    stdexec::inplace_stop_source first_stop;
    stdexec::inplace_stop_source second_stop;
    terminal_counts first_counts;
    terminal_counts second_counts;
    auto first = stdexec::connect(vkexec::detail::submit_fence(recorded_scope(*ctx)),
      recording_receiver{ .counts = &first_counts, .token = first_stop.get_token() });
    auto second = stdexec::connect(vkexec::detail::submit_fence(recorded_scope(*ctx)),
      recording_receiver{ .counts = &second_counts, .token = second_stop.get_token() });
    stdexec::start(first);
    stdexec::start(second);
    ctx.reset();
    first_stop.request_stop();
    bool const first_completed_early = first_counts.completions.load(std::memory_order_acquire) != 0;
    bool const second_completed_early = second_counts.completions.load(std::memory_order_acquire) != 0;

    VkSemaphoreSignalInfo signal_info{};
    signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signal_info.semaphore = gate;
    signal_info.value = k_release_value;
    VkResult const signaled = signal_semaphore(device, &signal_info);
    bool const first_completed = first_counts.wait();
    bool const second_completed = second_counts.wait();
    VkResult const blocker_wait = vkWaitForFences(device, 1, &blocker_fence, VK_TRUE, UINT64_MAX);
    blocker.release();
    vkDestroyFence(device, blocker_fence, nullptr);
    vkDestroySemaphore(device, gate, nullptr);
    REQUIRE(signaled == VK_SUCCESS);
    REQUIRE(blocker_wait == VK_SUCCESS);
    REQUIRE_FALSE(first_completed_early);
    REQUIRE_FALSE(second_completed_early);
    REQUIRE(first_completed);
    REQUIRE(second_completed);
    REQUIRE(first_counts.stopped.load() == k_one_completion);
    REQUIRE(second_counts.values.load() == k_one_completion);
    REQUIRE(first_counts.exactly_one());
    REQUIRE(second_counts.exactly_one());
  }
  retained.reset();
  REQUIRE(retired.wait_for(k_completion_timeout) == std::future_status::ready);
}

TEST_CASE("submit_fence reports injected queue submission failure once", "[vkexec][gpu][failure]")
{
  auto ctx = vkexec::test::require_context();
  VkResult injected = VK_ERROR_DEVICE_LOST;
  SECTION("device lost") { injected = VK_ERROR_DEVICE_LOST; }
  SECTION("host memory exhausted") { injected = VK_ERROR_OUT_OF_HOST_MEMORY; }
  injected_submit_result.store(injected, std::memory_order_relaxed);
  vkexec::detail::test_vulkan_dispatch dispatch;
  dispatch.queue_submit = failing_submit;
  dispatch.queue_submit2 = failing_submit2;
  dispatch_reset const installed{ dispatch, *ctx };

  stdexec::inplace_stop_source source;
  terminal_counts counts;
  auto operation = stdexec::connect(vkexec::detail::submit_fence(recorded_scope(*ctx)),
    recording_receiver{ .counts = &counts, .token = source.get_token() });
  stdexec::start(operation);
  REQUIRE(counts.wait());
  REQUIRE(counts.errors.load() == k_one_completion);
  REQUIRE(counts.error_code.load() == injected);
  REQUIRE(counts.exactly_one());
}

TEST_CASE("legacy and synchronization2 submission report injected Vulkan errors", "[vkexec][gpu][failure]")
{
  auto ctx = vkexec::test::require_context();
  vkexec::detail::synchronization_backend backend = vkexec::detail::synchronization_backend::legacy;
  SECTION("legacy") { backend = vkexec::detail::synchronization_backend::legacy; }
  SECTION("synchronization2")
  {
    if (!ctx->capabilities().synchronization2) { SKIP("Synchronization2 is unavailable"); }
    backend = vkexec::detail::select_synchronization_backend(ctx->capabilities());
  }
  injected_submit_result.store(VK_ERROR_DEVICE_LOST, std::memory_order_relaxed);
  vkexec::detail::test_vulkan_dispatch dispatch;
  dispatch.queue_submit = failing_submit;
  dispatch.queue_submit2 = failing_submit2;
  dispatch_reset const installed{ dispatch, *ctx };

  auto scope = recorded_scope(*ctx);
  std::array<VkCommandBuffer, 1> const commands{ scope.cmd };
  vkexec::detail::queue_synchronization_state queue_state;
  auto const submitted = vkexec::detail::submit(vkexec::queue_submit{ .command_buffers = commands },
    ctx->compute_queue(),
    backend,
    ctx->capabilities().timeline_semaphore,
    ctx->procs(),
    queue_state);
  REQUIRE_FALSE(submitted.has_value());
  REQUIRE(submitted.error().code.value() == VK_ERROR_DEVICE_LOST);
}

TEST_CASE("submit_fence reports injected fence wait failure after retirement", "[vkexec][gpu][failure]")
{
  auto ctx = vkexec::test::require_context();
  VkResult injected = VK_ERROR_DEVICE_LOST;
  SECTION("device lost") { injected = VK_ERROR_DEVICE_LOST; }
  SECTION("host memory exhausted") { injected = VK_ERROR_OUT_OF_HOST_MEMORY; }
  wait_failures.store(0, std::memory_order_relaxed);
  injected_wait_result.store(injected, std::memory_order_relaxed);
  vkexec::detail::test_vulkan_dispatch dispatch;
  dispatch.wait_for_fences = fail_wait_once;
  dispatch_reset const installed{ dispatch, *ctx };

  stdexec::inplace_stop_source source;
  terminal_counts counts;
  auto operation = stdexec::connect(vkexec::detail::submit_fence(recorded_scope(*ctx)),
    recording_receiver{ .counts = &counts, .token = source.get_token() });
  stdexec::start(operation);
  REQUIRE(counts.wait());
  REQUIRE(counts.errors.load() == k_one_completion);
  REQUIRE(counts.error_code.load() == injected);
  REQUIRE(counts.exactly_one());
}

TEST_CASE("submit_and_wait reports an injected wait failure after GPU retirement", "[vkexec][gpu][failure]")
{
  auto ctx = vkexec::test::require_context();
  wait_failures.store(0, std::memory_order_relaxed);
  injected_wait_result.store(VK_ERROR_DEVICE_LOST, std::memory_order_relaxed);
  vkexec::detail::test_vulkan_dispatch dispatch;
  dispatch.wait_for_fences = fail_wait_once;
  dispatch_reset const installed{ dispatch, *ctx };

  stdexec::inplace_stop_source source;
  terminal_counts counts;
  auto operation = stdexec::connect(vkexec::detail::submit_and_wait(recorded_scope(*ctx)),
    recording_receiver{ .counts = &counts, .token = source.get_token() });
  stdexec::start(operation);
  REQUIRE(counts.wait());
  REQUIRE(counts.errors.load() == k_one_completion);
  REQUIRE(counts.error_code.load() == VK_ERROR_DEVICE_LOST);
  REQUIRE(counts.exactly_one());

  vkexec::detail::set_test_vulkan_dispatch(nullptr);
  auto retry = vkexec::test::sync_wait_sender(vkexec::detail::submit_and_wait(recorded_scope(*ctx)));
  REQUIRE(vkexec::test::sync_wait_completed(retry));
}

TEST_CASE("submit_fence reports injected fence status failure after retirement", "[vkexec][gpu][failure]")
{
  auto ctx = vkexec::test::require_context();
  vkexec::detail::test_vulkan_dispatch dispatch;
  dispatch.get_fence_status = fail_fence_status;
  dispatch_reset const installed{ dispatch, *ctx };

  stdexec::inplace_stop_source source;
  terminal_counts counts;
  auto operation = stdexec::connect(vkexec::detail::submit_fence(recorded_scope(*ctx)),
    recording_receiver{ .counts = &counts, .token = source.get_token() });
  stdexec::start(operation);
  REQUIRE(counts.wait());
  REQUIRE(counts.errors.load() == k_one_completion);
  REQUIRE(counts.error_code.load() == VK_ERROR_DEVICE_LOST);
  REQUIRE(counts.exactly_one());
}

TEST_CASE("completion waiter reports the exact queue idle failure", "[vkexec][gpu][failure]")
{
  auto ctx = vkexec::test::require_context();
  vkexec::detail::test_vulkan_dispatch dispatch;
  dispatch.queue_wait_idle = fail_queue_wait_idle;
  dispatch_reset const installed{ dispatch, *ctx };
  stdexec::inplace_stop_source source;
  terminal_counts counts;
  auto enqueued = ctx->enqueue_fence_wait(VK_NULL_HANDLE,
    VK_NULL_HANDLE,
    source.get_token(),
    [&counts](std::optional<vkexec::error> failure, bool /*stopped*/) noexcept -> void {
      if (failure) {
        counts.error_code.store(failure->code.value(), std::memory_order_relaxed);
        counts.signal(counts.errors);
      } else {
        counts.signal(counts.values);
      }
    });
  REQUIRE(enqueued.has_value());
  REQUIRE(counts.wait());
  REQUIRE(counts.errors.load() == k_one_completion);
  REQUIRE(counts.error_code.load() == VK_ERROR_DEVICE_LOST);
  REQUIRE(counts.exactly_one());
}

TEST_CASE("static pass graph reports a recording failure once and skips later steps", "[vkexec][gpu][failure]")
{
  auto ctx = vkexec::test::require_context();
  int first = 0;
  int failing = 0;
  int skipped = 0;
  vkexec::pass_graph_sender<failing_record_step, failing_record_step, failing_record_step> graph{
    .state = vkexec::detail::context_access::state(*ctx),
    .steps =
      std::tuple{
        failing_record_step{ .recorded = &first },
        failing_record_step{ .recorded = &failing, .fail = true },
        failing_record_step{ .recorded = &skipped },
      },
    .step_queues = {},
    .step_predecessors = {},
    .presentation = {},
    .current_queue = {},
  };
  stdexec::inplace_stop_source source;
  terminal_counts counts;
  auto operation =
    stdexec::connect(std::move(graph), recording_receiver{ .counts = &counts, .token = source.get_token() });
  stdexec::start(operation);
  REQUIRE(counts.wait());
  REQUIRE(counts.errors.load() == k_one_completion);
  REQUIRE(counts.exactly_one());
  REQUIRE(first == k_one_completion);
  REQUIRE(failing == k_one_completion);
  REQUIRE(skipped == 0);
  auto retry = vkexec::test::sync_wait_sender(vkexec::detail::submit_and_wait(recorded_scope(*ctx)));
  REQUIRE(vkexec::test::sync_wait_completed(retry));
}

TEST_CASE("dynamic pass graph reports a recording failure once and skips later steps", "[vkexec][gpu][failure]")
{
  auto ctx = vkexec::test::require_context();
  int first = 0;
  int failing = 0;
  int skipped = 0;
  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  graph.append(vkexec::make_pass_adaptor(failing_record_step{ .recorded = &first }));
  graph.append(vkexec::make_pass_adaptor(failing_record_step{ .recorded = &failing, .fail = true }));
  graph.append(vkexec::make_pass_adaptor(failing_record_step{ .recorded = &skipped }));
  stdexec::inplace_stop_source source;
  terminal_counts counts;
  auto operation =
    stdexec::connect(std::move(graph), recording_receiver{ .counts = &counts, .token = source.get_token() });
  stdexec::start(operation);
  REQUIRE(counts.wait());
  REQUIRE(counts.errors.load() == k_one_completion);
  REQUIRE(counts.exactly_one());
  REQUIRE(first == k_one_completion);
  REQUIRE(failing == k_one_completion);
  REQUIRE(skipped == 0);
  auto retry = vkexec::test::sync_wait_sender(vkexec::detail::submit_and_wait(recorded_scope(*ctx)));
  REQUIRE(vkexec::test::sync_wait_completed(retry));
}

TEST_CASE("dynamic DAG retires an accepted branch when another submit fails", "[vkexec][gpu][failure]")
{
  auto ctx = vkexec::test::require_context();
  auto const compute = ctx->compute_queue_ref();
  auto const graphics = ctx->graphics_queue_ref();
  vkexec::queue_ref second_queue = compute;
  SECTION("single queue") { second_queue = compute; }
  SECTION("distinct queues")
  {
    if (graphics.queue == VK_NULL_HANDLE || vkexec::detail::same_queue(graphics, compute)) {
      SKIP("a distinct graphics queue is unavailable");
    }
    second_queue = graphics;
  }

  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  graph.append_queue(compute);
  (void)graph.append_after({}, vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {}));
  graph.append_queue(second_queue);
  auto const failing_branch =
    graph.append_after({}, vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {}));
  std::array const dependency{ failing_branch };
  graph.append_queue(compute);
  (void)graph.append_after(dependency, vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {}));

  branch_submit_calls.store(0, std::memory_order_relaxed);
  branch_accepted_submits.store(0, std::memory_order_relaxed);
  branch_rejected_submits.store(0, std::memory_order_relaxed);
  branch_real_submit2 = submit2_proc(*ctx);
  vkexec::detail::test_vulkan_dispatch dispatch;
  dispatch.queue_submit = fail_second_submit;
  dispatch.queue_submit2 = branch_real_submit2 ? fail_second_submit2 : nullptr;
  dispatch_reset const installed{ dispatch, *ctx };
  stdexec::inplace_stop_source source;
  terminal_counts counts;
  auto operation =
    stdexec::connect(std::move(graph), recording_receiver{ .counts = &counts, .token = source.get_token() });
  stdexec::start(operation);

  REQUIRE(counts.wait());
  REQUIRE(counts.errors.load() == k_one_completion);
  REQUIRE(counts.error_code.load() == VK_ERROR_DEVICE_LOST);
  REQUIRE(counts.exactly_one());
  REQUIRE(branch_submit_calls.load() == k_expected_branch_submits);
  REQUIRE(branch_accepted_submits.load() == k_one_completion);
  REQUIRE(branch_rejected_submits.load() == k_one_completion);
}

#if VKEXEC_HAS_EXCEPTIONS
TEST_CASE("submit_fence reclaims submitted work when waiter enqueue throws", "[vkexec][gpu][failure]")
{
  auto ctx = vkexec::test::require_context();
  vkexec::detail::test_vulkan_dispatch dispatch;
  dispatch.fail_enqueue = fail_enqueue;
  dispatch_reset const installed{ dispatch, *ctx };

  stdexec::inplace_stop_source source;
  terminal_counts counts;
  auto operation = stdexec::connect(vkexec::detail::submit_fence(recorded_scope(*ctx)),
    recording_receiver{ .counts = &counts, .token = source.get_token() });
  stdexec::start(operation);
  REQUIRE(counts.wait());
  REQUIRE(counts.errors.load() == k_one_completion);
  REQUIRE(counts.error_code.load() == static_cast<int>(vkexec::errc::unexpected_exception));
  REQUIRE(counts.exactly_one());
}
#endif

TEST_CASE("concurrent submit_fence cancellation produces one completion per operation", "[vkexec][gpu][stress]")
{
  auto ctx = vkexec::test::require_context();
  auto const submit2 = submit2_proc(*ctx);
  normal_submit2.store(submit2, std::memory_order_release);
  vkexec::detail::test_vulkan_dispatch dispatch;
  dispatch.queue_submit = checkpoint_submit;
  dispatch.queue_submit2 = submit2 != nullptr ? checkpoint_submit2 : nullptr;
  dispatch_reset const installed{ dispatch, *ctx };
  int const iterations = environment_number("VKEXEC_GPU_STRESS_ITERATIONS", k_default_stress_iterations);
  std::uint32_t const seed = environment_number("VKEXEC_GPU_STRESS_SEED", k_default_stress_seed);
  std::barrier start{ k_stress_threads };
  std::atomic<int> passed{ 0 };
  std::array<std::thread, k_stress_threads> submitters;

  for (int index = 0; index < k_stress_threads; ++index) {
    submitters.at(static_cast<std::size_t>(index)) = std::thread([&, index]() -> void {
      std::uint32_t random = seed + static_cast<std::uint32_t>(index);
      start.arrive_and_wait();
      for (int iteration = 0; iteration < iterations; ++iteration) {
        random = random * k_random_multiplier + k_random_increment;
        auto opened = vkexec::detail::submit_scope::open(*ctx);
        if (!opened) { return; }
        auto scope = vkexec::expected_take(opened);
        if (!scope.end_recording()) { return; }
        stdexec::inplace_stop_source source;
        terminal_counts counts;
        auto operation = stdexec::connect(vkexec::detail::submit_fence(std::move(scope)),
          recording_receiver{ .counts = &counts, .token = source.get_token() });
        std::uint32_t const mode = random % k_cancellation_modes;
        if (mode == 0U) { source.request_stop(); }
        if (mode == 1U) {
          submit_gate gate;
          gate.real_submit2 = submit2;
          std::thread starter{ [&operation, &gate]() -> void {
            thread_gate = &gate;
            stdexec::start(operation);
            thread_gate = nullptr;
          } };
          bool const submitted = gate.submitted.try_acquire_for(k_completion_timeout);
          source.request_stop();
          gate.release.release();
          starter.join();
          if (!submitted) { std::terminate(); }
        } else {
          int submit_calls = 0;
          thread_submit_calls = &submit_calls;
          stdexec::start(operation);
          thread_submit_calls = nullptr;
          if (mode == 0U && submit_calls != 0) { return; }
        }
        if (!counts.wait()) { std::terminate(); }
        if (mode == 0U && counts.stopped.load() != k_one_completion) { return; }
        if (mode == 1U && counts.stopped.load() != k_one_completion) { return; }
        if (mode == 2U) {
          int const values_before_stop = counts.values.load();
          source.request_stop();
          if (counts.values.load() != values_before_stop || counts.completions.load() != k_one_completion) { return; }
        }
        if (!counts.exactly_one() || counts.errors.load() != 0) { return; }
        passed.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto &submitter : submitters) { submitter.join(); }
  REQUIRE(passed.load() == k_stress_threads * iterations);
  auto retry = vkexec::test::sync_wait_sender(vkexec::detail::submit_and_wait(recorded_scope(*ctx)));
  REQUIRE(vkexec::test::sync_wait_completed(retry));
}

// NOLINTEND(readability-function-cognitive-complexity)
