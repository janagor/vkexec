// NOLINTBEGIN(readability-function-cognitive-complexity)

#include "gpu_failure_test_support.hpp"

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
