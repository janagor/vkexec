// NOLINTBEGIN(readability-function-cognitive-complexity)

#include "scheduler_test_support.hpp"
#include "test_helpers.hpp"
#include "vkexec/config.hpp"
#include "vkexec/context.hpp"
#include "vkexec/detail/object_synchronization.hpp"
#include "vkexec/detail/pass_types.hpp"
#include "vkexec/error.hpp"
#include "vkexec/pass.hpp"
#include "vkexec/result.hpp"
#include "vkexec/scheduler.hpp"
#include "vkexec/submit_scope.hpp"
#include <array>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <semaphore>
#include <stdexcept>
#include <stdexec/execution.hpp>
#include <stdexec/stop_token.hpp>
#include <thread>
#include <utility>
#include <vulkan/vulkan_core.h>

TEST_CASE("schedule_sender advertises completion scheduler", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto const sender = sched.schedule();

  auto const completion = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sender));
  REQUIRE(completion == sched);
  REQUIRE(!vkexec::detail::scheduler_access::state(completion));
}

TEST_CASE("null schedule observes pre-requested stop", "[vkexec][scheduler]")
{
  stdexec::inplace_stop_source source;
  source.request_stop();
  auto sender = stdexec::write_env(
    stdexec::schedule(vkexec::scheduler{ nullptr }), stdexec::prop{ stdexec::get_stop_token, source.get_token() });
  auto const waited = vkexec::test::sync_wait_sender(std::move(sender));
  REQUIRE(vkexec::test::sync_wait_stopped(waited));
}

TEST_CASE("null schedule reports invalid argument", "[vkexec][scheduler]")
{
  auto outcome = vkexec::test::sync_wait_sender(stdexec::schedule(vkexec::scheduler{ nullptr }));
  REQUIRE(outcome.failed());
  REQUIRE(outcome.error.value_or(vkexec::error{}).code == vkexec::make_error_code(vkexec::errc::invalid_argument));
}

TEST_CASE("null static pass graph reports invalid argument", "[vkexec][scheduler][pass]")
{
  auto graph = stdexec::schedule(vkexec::scheduler{ nullptr })
               | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 });
  auto outcome = vkexec::test::sync_wait_sender(graph);
  REQUIRE(outcome.failed());
  REQUIRE(outcome.error.value_or(vkexec::error{}).code == vkexec::make_error_code(vkexec::errc::invalid_argument));
}

TEST_CASE("null dynamic pass graph reports invalid argument", "[vkexec][scheduler][pass]")
{
  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(vkexec::scheduler{ nullptr }));
  graph.append(vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 }));
  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(outcome.failed());
  REQUIRE(outcome.error.value_or(vkexec::error{}).code == vkexec::make_error_code(vkexec::errc::invalid_argument));
}

TEST_CASE("null submit scope reports invalid argument", "[vkexec][scheduler][pass]")
{
  auto outcome = vkexec::test::sync_wait_sender(vkexec::detail::enter_submit_scope(nullptr));
  REQUIRE(outcome.failed());
  REQUIRE(outcome.error.value_or(vkexec::error{}).code == vkexec::make_error_code(vkexec::errc::invalid_argument));
}

TEST_CASE("scheduler retains runtime after context destruction", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto sched = ctx->get_scheduler();
  std::weak_ptr<vkexec::detail::context_state> const runtime = vkexec::detail::context_access::state(*ctx);
  REQUIRE(sched == ctx->get_scheduler());
  ctx.reset();
  REQUIRE(!runtime.expired());
  auto outcome = vkexec::test::sync_wait_sender(stdexec::schedule(sched));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
  sched = vkexec::scheduler{ nullptr };
  REQUIRE(runtime.expired());
}

TEST_CASE("schedule sender and connected operation retain runtime", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto sender = stdexec::schedule(ctx->get_scheduler());
  std::weak_ptr<vkexec::detail::context_state> const runtime = vkexec::detail::context_access::state(*ctx);
  std::promise<void> completed;
  auto ready = completed.get_future();
  ctx.reset();
  REQUIRE(!runtime.expired());
  auto operation = stdexec::connect(std::move(sender), completion_probe_receiver{ .completed = &completed });
  REQUIRE(!runtime.expired());
  stdexec::start(operation);
  REQUIRE(ready.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
}

TEST_CASE("final runtime release on host agent joins safely", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto retired = vkexec::detail::retirement_future(vkexec::detail::context_access::state(*ctx));
  std::promise<void> completed;
  auto ready = completed.get_future();
  auto retained = vkexec::detail::context_access::state(*ctx);
  auto enqueued = ctx->enqueue_host([state = std::move(retained), &completed]() noexcept -> void {
    (void)state;
    signal_promise(completed);
  });
  REQUIRE(enqueued.has_value());
  ctx.reset();
  REQUIRE(ready.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
  REQUIRE(retired.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
}

TEST_CASE("final runtime release on completion agent joins safely", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto retired = vkexec::detail::retirement_future(vkexec::detail::context_access::state(*ctx));
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VkFence fence{ VK_NULL_HANDLE };
  REQUIRE(vkCreateFence(ctx->device(), &fence_info, nullptr, &fence) == VK_SUCCESS);

  stdexec::inplace_stop_source const source;
  std::promise<void> completed;
  auto ready = completed.get_future();
  auto retained = vkexec::detail::context_access::state(*ctx);
  auto enqueued = ctx->enqueue_fence_wait(VK_NULL_HANDLE,
    fence,
    source.get_token(),
    [state = std::move(retained), &completed](
      std::optional<vkexec::error> const & /*failure*/, bool /*stopped*/) noexcept -> void {
      (void)state;
      signal_promise(completed);
    });
  REQUIRE(enqueued.has_value());
  ctx.reset();
  REQUIRE(ready.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
  REQUIRE(retired.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
}

TEST_CASE("static pass graph runs after context destruction", "[vkexec][scheduler][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto graph =
    stdexec::schedule(ctx->get_scheduler()) | vkexec::make_pass_adaptor(make_empty_pass_step([]() noexcept -> void {}));
  ctx.reset();
  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
}

TEST_CASE("dynamic pass graph runs after context destruction", "[vkexec][scheduler][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  graph.append(vkexec::make_pass_adaptor(make_empty_pass_step([]() noexcept -> void {})));
  ctx.reset();
  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
}

TEST_CASE("submit scope releases resources after context destruction", "[vkexec][scheduler][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(opened.has_value());
  auto scope = vkexec::expected_take(opened);
  REQUIRE(scope.end_recording().has_value());
  ctx.reset();
  auto outcome = vkexec::test::sync_wait_sender(vkexec::detail::submit_and_wait(std::move(scope)));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
}

TEST_CASE("schedule observes pre-requested stop", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  stdexec::inplace_stop_source source;
  source.request_stop();
  auto sender = stdexec::write_env(
    stdexec::schedule(ctx->get_scheduler()), stdexec::prop{ stdexec::get_stop_token, source.get_token() });
  auto const waited = vkexec::test::sync_wait_sender(std::move(sender));
  REQUIRE(vkexec::test::sync_wait_stopped(waited));
}

TEST_CASE("schedule observes stop while queued", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  std::promise<void> entered;
  auto entered_future = entered.get_future();
  std::atomic_bool release{ false };
  release_flag release_gate{ release };
  auto blocked = ctx->enqueue_host([&]() noexcept -> void {
    signal_promise(entered);
    release.wait(false, std::memory_order_acquire);
  });
  REQUIRE(blocked.has_value());
  auto const entered_status = entered_future.wait_for(std::chrono::seconds(5));
  if (entered_status != std::future_status::ready) { release_gate.release(); }
  REQUIRE(entered_status == std::future_status::ready);

  stdexec::inplace_stop_source source;
  std::atomic<int> outcome{ 0 };
  std::binary_semaphore completed{ 0 };
  auto sender = stdexec::write_env(
    stdexec::schedule(ctx->get_scheduler()), stdexec::prop{ stdexec::get_stop_token, source.get_token() });
  auto operation = stdexec::connect(std::move(sender),
    schedule_outcome_receiver{
      .outcome = &outcome,
      .completed = &completed,
    });
  stdexec::start(operation);
  source.request_stop();
  release_gate.release();

  REQUIRE(completed.try_acquire_for(std::chrono::seconds(5)));
  REQUIRE(outcome.load() == 3);
}

TEST_CASE("submit_and_wait releases a pre-stopped scope", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(opened.has_value());

  stdexec::inplace_stop_source source;
  source.request_stop();
  auto stopped_sender = stdexec::write_env(vkexec::detail::submit_and_wait(vkexec::expected_take(opened)),
    stdexec::prop{ stdexec::get_stop_token, source.get_token() });
  auto const stopped = vkexec::test::sync_wait_sender(std::move(stopped_sender));
  REQUIRE(vkexec::test::sync_wait_stopped(stopped));

  auto next = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(next.has_value());
  auto next_scope = vkexec::expected_take(next);
  REQUIRE(next_scope.end_recording().has_value());
  auto const retried = vkexec::test::sync_wait_sender(vkexec::detail::submit_and_wait(std::move(next_scope)));
  REQUIRE(vkexec::test::sync_wait_completed(retried));
}

TEST_CASE("compute submit scopes check out distinct pools and reuse released pools", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto first_result = vkexec::detail::submit_scope::open(*ctx);
  auto second_result = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(first_result.has_value());
  REQUIRE(second_result.has_value());
  auto first = vkexec::expected_take(first_result);
  auto second = vkexec::expected_take(second_result);
  REQUIRE(first.pool != second.pool);
  auto *const released = first.pool;
  REQUIRE(first.end_recording().has_value());
  first.release();
  auto third_result = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(third_result.has_value());
  auto third = vkexec::expected_take(third_result);
  REQUIRE(third.pool == released);
  REQUIRE(second.end_recording().has_value());
  REQUIRE(third.end_recording().has_value());
}

TEST_CASE("compute submit scopes record concurrently", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  std::atomic<int> completed{ 0 };
  auto record = [&]() -> void {
    auto opened = vkexec::detail::submit_scope::open(*ctx);
    if (!opened) { return; }
    auto scope = vkexec::expected_take(opened);
    if (!scope.end_recording()) { return; }
    auto outcome = vkexec::test::sync_wait_sender(vkexec::detail::submit_and_wait(std::move(scope)));
    if (vkexec::test::sync_wait_completed(outcome)) { completed.fetch_add(1); }
  };
  std::thread first(record);
  std::thread second(record);
  first.join();
  second.join();
  REQUIRE(completed.load() == 2);
}

TEST_CASE("direct command buffers record concurrently with exclusive pools", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  std::atomic<int> completed{ 0 };
  auto record = [&]() -> void {
    auto allocated = ctx->allocate_command_buffer();
    if (!allocated) { return; }
    VkCommandBuffer cmd = vkexec::expected_take(allocated);
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS && vkEndCommandBuffer(cmd) == VK_SUCCESS
        && ctx->submit_and_wait(cmd).has_value()) {
      completed.fetch_add(1);
    }
    ctx->free_command_buffer(cmd);
  };
  std::thread first(record);
  std::thread second(record);
  first.join();
  second.join();
  REQUIRE(completed.load() == 2);
}

TEST_CASE("queue guard move assignment releases the old state after its context is gone", "[vkexec][scheduler][gpu]")
{
  auto first_context = vkexec::test::require_context();
  auto second_context = vkexec::test::require_context();
  // Null keys are only used to exercise guard ownership; no Vulkan queue call
  // is made while either context is destroyed.
  auto old_guard = first_context->lock_queue(VK_NULL_HANDLE);
  auto replacement = second_context->lock_queue(VK_NULL_HANDLE);
  first_context.reset();
  second_context.reset();
  old_guard = std::move(replacement);
  SUCCEED();
}

TEST_CASE("descriptor sets allocate concurrently from distinct pools", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  VkDescriptorSetLayoutBinding binding{};
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = 1;
  layout_info.pBindings = &binding;
  VkDescriptorSetLayout layout{ VK_NULL_HANDLE };
  REQUIRE(vkCreateDescriptorSetLayout(ctx->device(), &layout_info, nullptr, &layout) == VK_SUCCESS);

  VkDescriptorPoolSize const size{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 };
  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &size;
  std::array<VkDescriptorPool, 2> pools{ VK_NULL_HANDLE, VK_NULL_HANDLE };
  for (VkDescriptorPool &pool : pools) {
    REQUIRE(vkCreateDescriptorPool(ctx->device(), &pool_info, nullptr, &pool) == VK_SUCCESS);
    REQUIRE(vkexec::detail::descriptor_pool_access::register_owned(*ctx, pool).has_value());
  }

  std::atomic<int> allocated{ 0 };
  auto allocate = [&](VkDescriptorPool pool) -> void {
    VkDescriptorSetAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.descriptorPool = pool;
    info.descriptorSetCount = 1;
    info.pSetLayouts = &layout;
    VkDescriptorSet set{ VK_NULL_HANDLE };
    auto const guard = vkexec::detail::descriptor_pool_access::lock(*ctx, pool);
    if (vkAllocateDescriptorSets(ctx->device(), &info, &set) == VK_SUCCESS) { allocated.fetch_add(1); }
  };
  std::thread first(allocate, pools.at(0));
  std::thread second(allocate, pools.at(1));
  first.join();
  second.join();

  for (VkDescriptorPool pool : pools) {
    auto const guard = vkexec::detail::descriptor_pool_access::lock(*ctx, pool);
    vkDestroyDescriptorPool(ctx->device(), pool, nullptr);
  }
  vkDestroyDescriptorSetLayout(ctx->device(), layout, nullptr);
  REQUIRE(allocated.load() == 2);
}

TEST_CASE("schedule completes on the context host agent", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();

  auto const caller = std::this_thread::get_id();
  auto const agent = ctx->host_agent_thread_id();
  REQUIRE(agent != caller);

  std::thread::id completed_on{};
  auto waited = vkexec::test::sync_wait_sender(stdexec::schedule(ctx->get_scheduler()) | stdexec::then([&]() -> void {
    completed_on = std::this_thread::get_id();
  }));
  REQUIRE(vkexec::test::sync_wait_completed(waited));

  REQUIRE(completed_on == agent);
  REQUIRE(completed_on != caller);
}

TEST_CASE("noexcept host task and nested enqueue run on the host agent", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto const agent = ctx->host_agent_thread_id();
  std::promise<void> completed;
  auto ready = completed.get_future();
  std::thread::id outer_thread{};
  std::thread::id inner_thread{};
  auto enqueued = ctx->enqueue_host([&]() noexcept -> void {
    outer_thread = std::this_thread::get_id();
    auto nested = ctx->enqueue_host([&]() noexcept -> void {
      inner_thread = std::this_thread::get_id();
      signal_promise(completed);
    });
    if (!nested) { signal_promise(completed); }
  });
  REQUIRE(enqueued.has_value());
  REQUIRE(ready.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
  REQUIRE(outer_thread == agent);
  REQUIRE(inner_thread == agent);
}

#if VKEXEC_HAS_EXCEPTIONS
TEST_CASE("throwing after_gpu completes with sender error", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  std::thread::id after_gpu_thread{};
  auto step = make_empty_pass_step([&]() -> void {
    after_gpu_thread = std::this_thread::get_id();
    throw std::runtime_error("after_gpu failed");
  });
  vkexec::pass_graph_sender<decltype(step)> graph{
    .state = vkexec::detail::context_access::state(*ctx),
    .steps = { step },
    .step_queues = {},
    .step_predecessors = {},
    .presentation = {},
    .current_queue = {},
  };

  auto outcome = vkexec::test::sync_wait_sender(graph);
  REQUIRE(outcome.failed());
  REQUIRE(outcome.error.value_or(vkexec::error{}).code == vkexec::make_error_code(vkexec::errc::unexpected_exception));
  REQUIRE(after_gpu_thread == ctx->host_agent_thread_id());
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(opened.has_value());
  auto scope = vkexec::expected_take(opened);
  REQUIRE(scope.end_recording().has_value());
  auto retry = vkexec::test::sync_wait_sender(vkexec::detail::submit_and_wait(std::move(scope)));
  REQUIRE(vkexec::test::sync_wait_completed(retry));
}

TEST_CASE("throwing dynamic after_gpu completes with sender error", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  std::thread::id after_gpu_thread{};
  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()))
               | vkexec::make_pass_adaptor(make_empty_pass_step([&]() -> void {
                   after_gpu_thread = std::this_thread::get_id();
                   throw std::runtime_error("after_gpu failed");
                 }));

  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(outcome.failed());
  REQUIRE(outcome.error.value_or(vkexec::error{}).code == vkexec::make_error_code(vkexec::errc::unexpected_exception));
  REQUIRE(after_gpu_thread == ctx->host_agent_thread_id());
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(opened.has_value());
  auto scope = vkexec::expected_take(opened);
  REQUIRE(scope.end_recording().has_value());
  auto retry = vkexec::test::sync_wait_sender(vkexec::detail::submit_and_wait(std::move(scope)));
  REQUIRE(vkexec::test::sync_wait_completed(retry));
}
#endif

TEST_CASE("starts_on runs the child on the context host agent", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();

  auto const caller = std::this_thread::get_id();
  auto const agent = ctx->host_agent_thread_id();

  std::thread::id ran_on{};
  auto waited = vkexec::test::sync_wait_sender(stdexec::starts_on(
    ctx->get_scheduler(), stdexec::just() | stdexec::then([&]() -> void { ran_on = std::this_thread::get_id(); })));
  REQUIRE(vkexec::test::sync_wait_completed(waited));

  REQUIRE(ran_on == agent);
  REQUIRE(ran_on != caller);
}


// NOLINTEND(readability-function-cognitive-complexity)
