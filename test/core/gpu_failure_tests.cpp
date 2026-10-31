// NOLINTBEGIN(readability-function-cognitive-complexity)

#include "gpu_failure_test_support.hpp"
#include <vkexec/record_into.hpp>

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


// NOLINTEND(readability-function-cognitive-complexity)


namespace {
std::atomic<int> recording_wait_calls{};
}

TEST_CASE("external graph recording performs no submissions or waits", "[vkexec][gpu][record_into]")
{
  auto ctx = vkexec::test::require_context();
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(opened);
  auto scope = vkexec::expected_take(opened);
  int submit_calls{};
  auto *submit2 = ctx->procs().queue_submit2;
  normal_submit2.store(submit2, std::memory_order_release);
  vkexec::detail::test_vulkan_dispatch dispatch;
  dispatch.queue_submit = checkpoint_submit;
  dispatch.queue_submit2 = submit2 ? checkpoint_submit2 : nullptr;
  recording_wait_calls.store(0);
  dispatch.wait_for_fences = [](VkDevice, std::uint32_t, VkFence const *, VkBool32, std::uint64_t) -> VkResult {
    ++recording_wait_calls;
    return VK_SUCCESS;
  };
  dispatch.queue_wait_idle = [](VkQueue) -> VkResult {
    ++recording_wait_calls;
    return VK_SUCCESS;
  };
  {
    dispatch_reset const installed{ dispatch, *ctx };
    thread_submit_calls = &submit_calls;
    auto graph = stdexec::schedule(ctx->get_scheduler()) | vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer) {});
    auto recorded = vkexec::record_into(*ctx, scope.cmd, std::move(graph), { scope.queue });
    thread_submit_calls = nullptr;
    REQUIRE(recorded);
    CHECK(submit_calls == 0);
    CHECK(recording_wait_calls.load() == 0);
    REQUIRE(vkResetCommandBuffer(scope.cmd, 0) == VK_SUCCESS);
    REQUIRE(recorded->abandon());
    CHECK(submit_calls == 0);
    CHECK(recording_wait_calls.load() == 0);
  }
}

TEST_CASE("single queue owned DAG submits one scope per prepared batch", "[vkexec][gpu][record_into]")
{
  auto ctx = vkexec::test::require_context();
  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  std::vector<std::size_t> incoming;
  for (std::size_t index{}; index < 3; ++index) {
    auto node = graph.append_after(incoming, vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer) {}));
    incoming = { node };
  }
  auto prepared = vkexec::detail::prepare_graph(
    *ctx, graph.steps, graph.step_queues, graph.step_predecessors, ctx->compute_queue_ref());
  REQUIRE(prepared);
  REQUIRE(vkexec::detail::terminal_batches(prepared->execution).size() == 1);
  auto *submit2 = ctx->procs().queue_submit2;
  normal_submit2.store(submit2, std::memory_order_release);
  vkexec::detail::test_vulkan_dispatch dispatch;
  dispatch.queue_submit = checkpoint_submit;
  dispatch.queue_submit2 = submit2 ? checkpoint_submit2 : nullptr;
  dispatch_reset const installed{ dispatch, *ctx };
  int submissions{};
  thread_submit_calls = &submissions;
  auto completed = vkexec::test::sync_wait_sender(std::move(graph));
  thread_submit_calls = nullptr;
  REQUIRE(vkexec::test::sync_wait_completed(completed));
  CHECK(submissions == static_cast<int>(prepared->execution.batches.size()));
  CHECK(submissions == 3);
}
