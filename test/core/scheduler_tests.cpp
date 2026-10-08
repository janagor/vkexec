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

namespace ex = stdexec;

namespace {

struct empty_receiver_env
{
};

auto signal_promise(std::promise<void> &promise) noexcept -> void
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
  using receiver_concept = ex::receiver_t;

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
  using receiver_concept = ex::receiver_t;

  std::atomic<int> *outcome{ nullptr };
  std::binary_semaphore *completed{ nullptr };

  auto complete(int value) const noexcept -> void
  {
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

}// namespace

template<class Sender, class CompletionTag>
concept advertises_completion_scheduler =
  requires(Sender const &sndr) { ex::get_completion_scheduler<CompletionTag>(ex::get_env(sndr)); };

using move_only_graph_t = vkexec::pass_graph_sender<move_only_pass_step>;

template<class Graph>
concept rvalue_connectable_graph =
  requires(Graph graph, completion_probe_receiver receiver) { ex::connect(std::move(graph), std::move(receiver)); };

template<class Graph>
concept const_lvalue_connectable_graph =
  requires(Graph const &graph, completion_probe_receiver receiver) { ex::connect(graph, std::move(receiver)); };

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
static_assert(advertises_completion_scheduler<vkexec::schedule_sender, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::schedule_sender, ex::set_error_t>);
static_assert(!advertises_completion_scheduler<vkexec::schedule_sender, ex::set_stopped_t>);
static_assert(std::same_as<vkexec::schedule_sender::completion_signatures,
  ex::completion_signatures<ex::set_value_t(), ex::set_error_t(vkexec::error), ex::set_stopped_t()>>);
static_assert(std::same_as<vkexec::detail::submit_and_wait_sender::completion_signatures,
  ex::completion_signatures<ex::set_value_t(), ex::set_error_t(vkexec::error), ex::set_stopped_t()>>);

static_assert(advertises_completion_scheduler<vkexec::pass_graph_sender<>, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::pass_graph_sender<>, ex::set_error_t>);

static_assert(!advertises_completion_scheduler<vkexec::detail::enter_submit_scope_sender, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::detail::submit_and_wait_sender, ex::set_value_t>);
static_assert(!advertises_completion_scheduler<vkexec::detail::submit_fence_sender, ex::set_value_t>);

static_assert(vkexec::vkexec_predecessor<vkexec::schedule_sender>);
static_assert(vkexec::vkexec_predecessor<vkexec::pass_graph_sender<>>);
static_assert(std::move_constructible<vkexec::dynamic_pass_graph_sender>);
static_assert(!std::copy_constructible<vkexec::dynamic_pass_graph_sender>);
static_assert(vkexec::detail::is_pass_graph_sender_v<vkexec::dynamic_pass_graph_sender>);
static_assert(vkexec::vkexec_predecessor<vkexec::dynamic_pass_graph_sender>);
TEST_CASE("schedule_sender advertises completion scheduler", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto const sender = sched.schedule();

  auto const completion = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sender));
  REQUIRE(completion == sched);
  REQUIRE(!vkexec::detail::scheduler_access::state(completion));
}

TEST_CASE("null schedule observes pre-requested stop", "[vkexec][scheduler]")
{
  ex::inplace_stop_source source;
  source.request_stop();
  auto sender =
    ex::write_env(ex::schedule(vkexec::scheduler{ nullptr }), ex::prop{ ex::get_stop_token, source.get_token() });
  auto const waited = vkexec::test::sync_wait_sender(std::move(sender));
  REQUIRE(vkexec::test::sync_wait_stopped(waited));
}

TEST_CASE("null schedule reports invalid argument", "[vkexec][scheduler]")
{
  auto outcome = vkexec::test::sync_wait_sender(ex::schedule(vkexec::scheduler{ nullptr }));
  REQUIRE(outcome.failed());
  REQUIRE(outcome.error.value_or(vkexec::error{}).code == vkexec::make_error_code(vkexec::errc::invalid_argument));
}

TEST_CASE("null static pass graph reports invalid argument", "[vkexec][scheduler][pass]")
{
  auto graph = ex::schedule(vkexec::scheduler{ nullptr })
               | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 });
  auto outcome = vkexec::test::sync_wait_sender(graph);
  REQUIRE(outcome.failed());
  REQUIRE(outcome.error.value_or(vkexec::error{}).code == vkexec::make_error_code(vkexec::errc::invalid_argument));
}

TEST_CASE("null dynamic pass graph reports invalid argument", "[vkexec][scheduler][pass]")
{
  auto graph = vkexec::make_dynamic_pass_graph(ex::schedule(vkexec::scheduler{ nullptr }));
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
  auto outcome = vkexec::test::sync_wait_sender(ex::schedule(sched));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
  sched = vkexec::scheduler{ nullptr };
  REQUIRE(runtime.expired());
}

TEST_CASE("schedule sender and connected operation retain runtime", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto sender = ex::schedule(ctx->get_scheduler());
  std::weak_ptr<vkexec::detail::context_state> const runtime = vkexec::detail::context_access::state(*ctx);
  std::promise<void> completed;
  auto ready = completed.get_future();
  ctx.reset();
  REQUIRE(!runtime.expired());
  auto operation = ex::connect(std::move(sender), completion_probe_receiver{ .completed = &completed });
  REQUIRE(!runtime.expired());
  ex::start(operation);
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

  ex::inplace_stop_source const source;
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
    ex::schedule(ctx->get_scheduler()) | vkexec::make_pass_adaptor(make_empty_pass_step([]() noexcept -> void {}));
  ctx.reset();
  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
}

TEST_CASE("dynamic pass graph runs after context destruction", "[vkexec][scheduler][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto graph = vkexec::make_dynamic_pass_graph(ex::schedule(ctx->get_scheduler()));
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
  ex::inplace_stop_source source;
  source.request_stop();
  auto sender = ex::write_env(ex::schedule(ctx->get_scheduler()), ex::prop{ ex::get_stop_token, source.get_token() });
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

  ex::inplace_stop_source source;
  std::atomic<int> outcome{ 0 };
  std::binary_semaphore completed{ 0 };
  auto sender = ex::write_env(ex::schedule(ctx->get_scheduler()), ex::prop{ ex::get_stop_token, source.get_token() });
  auto operation = ex::connect(std::move(sender),
    schedule_outcome_receiver{
      .outcome = &outcome,
      .completed = &completed,
    });
  ex::start(operation);
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

  ex::inplace_stop_source source;
  source.request_stop();
  auto stopped_sender = ex::write_env(
    vkexec::detail::submit_and_wait(vkexec::expected_take(opened)), ex::prop{ ex::get_stop_token, source.get_token() });
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

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("schedule completes on the context host agent", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();

  auto const caller = std::this_thread::get_id();
  auto const agent = ctx->host_agent_thread_id();
  REQUIRE(agent != caller);

  std::thread::id completed_on{};
  auto waited = vkexec::test::sync_wait_sender(
    ex::schedule(ctx->get_scheduler()) | ex::then([&]() -> void { completed_on = std::this_thread::get_id(); }));
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
}

TEST_CASE("throwing dynamic after_gpu completes with sender error", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  std::thread::id after_gpu_thread{};
  auto graph = vkexec::make_dynamic_pass_graph(ex::schedule(ctx->get_scheduler()))
               | vkexec::make_pass_adaptor(make_empty_pass_step([&]() -> void {
                   after_gpu_thread = std::this_thread::get_id();
                   throw std::runtime_error("after_gpu failed");
                 }));

  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(outcome.failed());
  REQUIRE(outcome.error.value_or(vkexec::error{}).code == vkexec::make_error_code(vkexec::errc::unexpected_exception));
  REQUIRE(after_gpu_thread == ctx->host_agent_thread_id());
}
#endif

TEST_CASE("starts_on runs the child on the context host agent", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();

  auto const caller = std::this_thread::get_id();
  auto const agent = ctx->host_agent_thread_id();

  std::thread::id ran_on{};
  auto waited = vkexec::test::sync_wait_sender(
    ex::starts_on(ctx->get_scheduler(), ex::just() | ex::then([&]() -> void { ran_on = std::this_thread::get_id(); })));
  REQUIRE(vkexec::test::sync_wait_completed(waited));

  REQUIRE(ran_on == agent);
  REQUIRE(ran_on != caller);
}

TEST_CASE("schedule_sender advertises completion domain", "[vkexec][scheduler][domain]")
{
  vkexec::scheduler_env const env{};
  STATIC_REQUIRE(std::same_as<decltype(env.query(ex::get_completion_domain_t<ex::set_value_t>{})), vkexec::domain>);
  STATIC_REQUIRE(std::same_as<decltype(env.query(ex::get_domain_t{})), vkexec::domain>);

  vkexec::scheduler const sched{ nullptr };
  STATIC_REQUIRE(std::same_as<decltype(sched.query(ex::get_completion_domain_t<ex::set_value_t>{})), vkexec::domain>);
  STATIC_REQUIRE(std::same_as<decltype(sched.query(ex::get_domain_t{})), vkexec::domain>);
}

TEST_CASE("domain-only sender environment advertises vkexec domain", "[vkexec][scheduler][domain]")
{
  vkexec::detail::domain_env const env{};

  STATIC_REQUIRE(std::same_as<decltype(env.query(ex::get_completion_domain_t<ex::set_value_t>{})), vkexec::domain>);
  STATIC_REQUIRE(std::same_as<decltype(env.query(ex::get_domain_t{})), vkexec::domain>);
}

TEST_CASE("semantic pass expression preserves predecessor value completion scheduler", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto pred = sched.schedule() | ex::then([]() -> void {});
  auto const sndr = pred | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 });

  STATIC_REQUIRE(ex::sender<decltype(sndr)>);
  auto const completion = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr));
  REQUIRE(completion == sched);
}

TEST_CASE("pass closures compose independently from a sender", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto pipeline = vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 })
                  | vkexec::barrier::compute_to_compute()
                  | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 });
  auto sndr = sched.schedule() | pipeline;
  auto second = sched.schedule() | pipeline;
  STATIC_REQUIRE(ex::sender<decltype(sndr)>);
  STATIC_REQUIRE(vkexec::detail::is_sender_expr_v<decltype(sndr)>);
  STATIC_REQUIRE(std::same_as<vkexec::detail::expression_tag_t<decltype(sndr)>, vkexec::compute_pass_t>);
  auto lowered = ex::transform_sender(sndr, ex::env<>{});
  auto second_lowered = ex::transform_sender(second, ex::env<>{});
  REQUIRE(!lowered.state);
  REQUIRE(!second_lowered.state);
}

TEST_CASE("compute pass direct and pipe forms produce the same sender", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto direct = vkexec::compute_pass(sched.schedule(), vkexec::compute_bind{}, vkexec::dispatch{});
  auto piped = sched.schedule() | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{});

  STATIC_REQUIRE(std::same_as<decltype(direct), decltype(piped)>);
  STATIC_REQUIRE(vkexec::detail::is_sender_expr_v<decltype(direct)>);
  REQUIRE(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(direct))
          == ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(piped)));
}

TEST_CASE("semantic pass expressions fuse into one typed graph", "[vkexec][scheduler]")
{
  vkexec::scheduler const sched{ nullptr };
  auto expr = sched.schedule() | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{})
              | vkexec::barrier::compute_to_compute()
              | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{});
  auto sndr = ex::transform_sender(expr, ex::env<>{});

  STATIC_REQUIRE(std::tuple_size_v<std::remove_cvref_t<decltype(sndr.steps)>> == 3);
  REQUIRE(std::get<0>(sndr.steps).bind.pipeline == VK_NULL_HANDLE);
}

TEST_CASE("semantic pass expressions materialize across a then boundary", "[vkexec][scheduler]")
{
  using expr_t = decltype(ex::schedule(std::declval<vkexec::scheduler>()) | ex::then([]() noexcept -> void {})
                          | vkexec::make_pass_adaptor(std::declval<counting_pass_step>())
                          | vkexec::make_pass_adaptor(std::declval<counting_pass_step>()));

  STATIC_REQUIRE(ex::sender<expr_t>);
  using lowered_t = decltype(ex::transform_sender(std::declval<expr_t>(), ex::env<>{}));
  STATIC_REQUIRE(ex::sender<lowered_t>);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("pass_graph_sender start returns before GPU completion", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();

  std::promise<void> after_gpu_entered;
  auto after_gpu_ready = after_gpu_entered.get_future();
  std::atomic_bool allow_completion{ false };
  release_flag release_gate{ allow_completion };
  std::promise<void> receiver_completed;
  auto receiver_done = receiver_completed.get_future();

  auto step = make_empty_pass_step([&]() noexcept -> void {
    signal_promise(after_gpu_entered);
    allow_completion.wait(false, std::memory_order_acquire);
  });
  vkexec::pass_graph_sender<decltype(step)> graph{
    .state = vkexec::detail::context_access::state(*ctx),
    .steps = { step },
    .step_queues = {},
    .step_predecessors = {},
    .presentation = {},
    .current_queue = {},
  };

  auto operation = ex::connect(graph, completion_probe_receiver{ .completed = &receiver_completed });

  std::promise<void> start_returned;
  auto start_done = start_returned.get_future();

  std::thread starter{ [&]() -> void {
    ex::start(operation);
    signal_promise(start_returned);
  } };

  auto const after_gpu_status = after_gpu_ready.wait_for(std::chrono::seconds(5));
  auto const start_status = start_done.wait_for(std::chrono::seconds(2));
  auto const receiver_status_before_release = receiver_done.wait_for(std::chrono::seconds(0));

  release_gate.release();
  auto const receiver_status_after_release = receiver_done.wait_for(std::chrono::seconds(5));
  starter.join();

  REQUIRE(after_gpu_status == std::future_status::ready);
  REQUIRE(start_status == std::future_status::ready);
  REQUIRE(receiver_status_before_release == std::future_status::timeout);
  REQUIRE(receiver_status_after_release == std::future_status::ready);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("dynamic pass graph start returns before GPU completion", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();

  std::promise<void> after_gpu_entered;
  auto after_gpu_ready = after_gpu_entered.get_future();
  std::atomic_bool allow_completion{ false };
  release_flag release_gate{ allow_completion };
  std::promise<void> receiver_completed;
  auto receiver_done = receiver_completed.get_future();

  auto graph = vkexec::make_dynamic_pass_graph(ex::schedule(ctx->get_scheduler()))
               | vkexec::make_pass_adaptor(make_empty_pass_step([&]() noexcept -> void {
                   signal_promise(after_gpu_entered);
                   allow_completion.wait(false, std::memory_order_acquire);
                 }));

  auto operation = ex::connect(std::move(graph), completion_probe_receiver{ .completed = &receiver_completed });

  std::promise<void> start_returned;
  auto start_done = start_returned.get_future();

  std::thread starter{ [&]() -> void {
    ex::start(operation);
    signal_promise(start_returned);
  } };

  auto const after_gpu_status = after_gpu_ready.wait_for(std::chrono::seconds(5));
  auto const start_status = start_done.wait_for(std::chrono::seconds(2));
  auto const receiver_status_before_release = receiver_done.wait_for(std::chrono::seconds(0));

  release_gate.release();
  auto const receiver_status_after_release = receiver_done.wait_for(std::chrono::seconds(5));
  starter.join();

  REQUIRE(after_gpu_status == std::future_status::ready);
  REQUIRE(start_status == std::future_status::ready);
  REQUIRE(receiver_status_before_release == std::future_status::timeout);
  REQUIRE(receiver_status_after_release == std::future_status::ready);
}

TEST_CASE("dynamic pass graph executes multiple runtime steps", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();

  int first_recorded = 0;
  int second_recorded = 0;
  int first_after_gpu = 0;
  int second_after_gpu = 0;

  auto first = vkexec::detail::make_callback_pass_step(
    [&](vkexec::context & /*unused*/,
      VkCommandBuffer /*unused*/,
      vkexec::detail::pass_cleanup & /*unused*/) -> vkexec::status {
      ++first_recorded;
      return {};
    },
    [&]() -> void { ++first_after_gpu; });

  auto second = vkexec::detail::make_callback_pass_step(
    [&](vkexec::context & /*unused*/,
      VkCommandBuffer /*unused*/,
      vkexec::detail::pass_cleanup & /*unused*/) -> vkexec::status {
      ++second_recorded;
      return {};
    },
    [&]() -> void { ++second_after_gpu; });

  auto graph = vkexec::make_dynamic_pass_graph(ex::schedule(ctx->get_scheduler())) | vkexec::make_pass_adaptor(first)
               | vkexec::barrier::compute_to_compute() | vkexec::make_pass_adaptor(second);

  auto result = vkexec::test::sync_wait_sender(std::move(graph));

  REQUIRE(vkexec::test::sync_wait_completed(result));
  REQUIRE(first_recorded == 1);
  REQUIRE(second_recorded == 1);
  REQUIRE(first_after_gpu == 1);
  REQUIRE(second_after_gpu == 1);
}

TEST_CASE("DAG fan-out and fan-in completes all submitted branches", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto graph = vkexec::make_dynamic_pass_graph(ex::schedule(ctx->get_scheduler()));
  auto const compute = ctx->compute_queue_ref();
  auto const graphics = ctx->graphics_queue_ref();
  auto make_empty_node = []() -> decltype(auto) {
    return vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*unused*/) -> void {});
  };

  graph.append_queue(compute);
  auto const root = graph.append_after({}, make_empty_node());
  std::array const root_dependency{ root };
  auto const left = graph.append_after(root_dependency, make_empty_node());
  if (graphics.queue != VK_NULL_HANDLE) { graph.append_queue(graphics); }
  auto const right = graph.append_after(root_dependency, make_empty_node());
  std::array const join_dependencies{ left, right };
  graph.append_queue(compute);
  (void)graph.append_after(join_dependencies, make_empty_node());

  auto const result = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(result));
}

TEST_CASE("DAG terminal join waits for independent queue branches", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto const compute = ctx->compute_queue_ref();
  auto const graphics = ctx->graphics_queue_ref();
  if (graphics.queue == VK_NULL_HANDLE || vkexec::detail::same_queue(compute, graphics)) {
    SKIP("Distinct graphics and compute queues are unavailable");
  }

  auto graph = vkexec::make_dynamic_pass_graph(ex::schedule(ctx->get_scheduler()));
  graph.append_queue(compute);
  auto const compute_branch =
    graph.append_after({}, vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*unused*/) -> void {}));
  graph.append_queue(graphics);
  auto const graphics_branch =
    graph.append_after({}, vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*unused*/) -> void {}));
  REQUIRE(compute_branch != graphics_branch);

  auto const result = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(result));
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("sync_wait still waits for pass_graph_sender completion", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();

  std::promise<void> after_gpu_entered;
  auto after_gpu_ready = after_gpu_entered.get_future();
  std::atomic_bool allow_completion{ false };
  release_flag release_gate{ allow_completion };
  std::atomic_bool sync_wait_returned{ false };
  std::atomic_bool completed_successfully{ false };

  auto step = make_empty_pass_step([&]() noexcept -> void {
    signal_promise(after_gpu_entered);
    allow_completion.wait(false, std::memory_order_acquire);
  });
  vkexec::pass_graph_sender<decltype(step)> graph{
    .state = vkexec::detail::context_access::state(*ctx),
    .steps = { step },
    .step_queues = {},
    .step_predecessors = {},
    .presentation = {},
    .current_queue = {},
  };

  std::thread waiter{ [&]() -> void {
    auto const waited = vkexec::test::sync_wait_sender(graph);
    completed_successfully.store(vkexec::test::sync_wait_completed(waited), std::memory_order_release);
    sync_wait_returned.store(true, std::memory_order_release);
  } };

  auto const after_gpu_status = after_gpu_ready.wait_for(std::chrono::seconds(5));
  auto const returned_while_blocked = sync_wait_returned.load(std::memory_order_acquire);

  release_gate.release();
  waiter.join();

  auto const returned_after_release = sync_wait_returned.load(std::memory_order_acquire);
  auto const success = completed_successfully.load(std::memory_order_acquire);

  REQUIRE(after_gpu_status == std::future_status::ready);
  REQUIRE_FALSE(returned_while_blocked);
  REQUIRE(returned_after_release);
  REQUIRE(success);
}

TEST_CASE("pass_graph_sender completes on the context host scheduler", "[vkexec][scheduler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto const agent = ctx->host_agent_thread_id();

  std::thread::id completed_on{};
  auto step = make_empty_pass_step([]() noexcept -> void {});
  vkexec::pass_graph_sender<decltype(step)> graph{
    .state = vkexec::detail::context_access::state(*ctx),
    .steps = { step },
    .step_queues = {},
    .step_predecessors = {},
    .presentation = {},
    .current_queue = {},
  };

  auto waited = vkexec::test::sync_wait_sender(
    std::move(graph) | ex::then([&]() noexcept -> void { completed_on = std::this_thread::get_id(); }));
  REQUIRE(vkexec::test::sync_wait_completed(waited));
  REQUIRE(completed_on == agent);
}

TEST_CASE("pass composition retains each concrete step type", "[vkexec][pass]")
{
  vkexec::scheduler sched{ nullptr };

  auto graph = ex::transform_sender(
    ex::schedule(sched) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 }), ex::env<>{});
  STATIC_REQUIRE(vkexec::detail::is_pass_graph_sender_v<decltype(graph)>);
  STATIC_REQUIRE(std::tuple_size_v<decltype(graph.steps)> == 1);

  auto graph2 = ex::transform_sender(std::move(graph) | vkexec::barrier::compute_to_compute(), ex::env<>{});
  STATIC_REQUIRE(vkexec::detail::is_pass_graph_sender_v<decltype(graph2)>);
  STATIC_REQUIRE(std::tuple_size_v<decltype(graph2.steps)> == 2);
  STATIC_REQUIRE(!std::same_as<decltype(graph), decltype(graph2)>);

  auto graph3 = ex::transform_sender(
    std::move(graph2) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 }), ex::env<>{});
  STATIC_REQUIRE(vkexec::detail::is_pass_graph_sender_v<decltype(graph3)>);
  STATIC_REQUIRE(std::tuple_size_v<decltype(graph3.steps)> == 3);
  STATIC_REQUIRE(!std::same_as<decltype(graph2), decltype(graph3)>);
  REQUIRE(!graph3.state);
}

// NOLINTBEGIN(bugprone-unchecked-optional-access)
TEST_CASE("dynamic pass graph keeps one sender type", "[vkexec][pass]")
{
  constexpr std::uint32_t k_queue_family = 2U;
  constexpr std::uint32_t k_next_queue_family = 3U;
  vkexec::scheduler sched{ nullptr };
  auto graph = vkexec::make_dynamic_pass_graph(ex::schedule(sched));
  auto const queue = vkexec::queue_ref{ .queue = VK_NULL_HANDLE, .family = k_queue_family };
  auto const next_queue = vkexec::queue_ref{ .queue = VK_NULL_HANDLE, .family = k_next_queue_family };

  graph.append(vkexec::on_queue(queue));
  REQUIRE(graph.steps.empty());

  graph.append(vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 }));
  STATIC_REQUIRE(std::same_as<decltype(graph), vkexec::dynamic_pass_graph_sender>);
  REQUIRE(graph.steps.size() == 1);
  REQUIRE(graph.step_queues.at(0).has_value());
  REQUIRE(graph.step_queues.at(0)->family == queue.family);

  graph.append(vkexec::barrier::compute_to_compute());
  REQUIRE(graph.steps.size() == 2);

  graph.append(vkexec::on_queue(next_queue));
  graph.append(vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{ .x = 1 }));
  REQUIRE(graph.steps.size() == 3);
  REQUIRE(graph.step_queues.size() == graph.steps.size());
  REQUIRE(graph.step_queues.at(1).has_value());
  REQUIRE(graph.step_queues.at(1)->family == queue.family);
  REQUIRE(graph.step_queues.at(2).has_value());
  REQUIRE(graph.step_queues.at(2)->family == next_queue.family);
}

TEST_CASE("queue affinity survives static graph composition", "[vkexec][pass]")
{
  constexpr std::uint32_t k_queue_family = 2U;
  vkexec::scheduler sched{ nullptr };
  auto const queue = vkexec::queue_ref{ .queue = VK_NULL_HANDLE, .family = k_queue_family };

  auto first = ex::transform_sender(
    ex::schedule(sched) | vkexec::on_queue(queue) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{}),
    ex::env<>{});
  auto second = ex::transform_sender(
    std::move(first) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{}), ex::env<>{});

  STATIC_REQUIRE(std::tuple_size_v<decltype(second.steps)> == 2);
  REQUIRE(second.step_queues.size() == std::tuple_size_v<decltype(second.steps)>);
  REQUIRE(second.step_queues.at(0).has_value());
  REQUIRE(second.step_queues.at(1).has_value());
  REQUIRE(second.step_queues.at(0)->family == queue.family);
  REQUIRE(second.step_queues.at(1)->family == queue.family);
}

TEST_CASE("on_queue marks subsequent static pass steps", "[vkexec][pass]")
{
  constexpr std::uint32_t k_graphics_family = 1U;
  constexpr std::uint32_t k_compute_family = 2U;
  vkexec::scheduler sched{ nullptr };
  auto const graphics = vkexec::queue_ref{ .queue = VK_NULL_HANDLE, .family = k_graphics_family };
  auto const compute = vkexec::queue_ref{ .queue = VK_NULL_HANDLE, .family = k_compute_family };

  auto graph = ex::transform_sender(
    ex::schedule(sched) | vkexec::on_queue(graphics) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{})
      | vkexec::on_queue(compute) | vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{}),
    ex::env<>{});

  STATIC_REQUIRE(std::tuple_size_v<decltype(graph.steps)> == 2);
  REQUIRE(graph.step_queues.size() == std::tuple_size_v<decltype(graph.steps)>);
  REQUIRE(graph.step_queues.at(0).has_value());
  REQUIRE(graph.step_queues.at(1).has_value());
  REQUIRE(graph.step_queues.at(0)->family == graphics.family);
  REQUIRE(graph.step_queues.at(1)->family == compute.family);
  REQUIRE(graph.current_queue.has_value());
  REQUIRE(graph.current_queue->family == compute.family);
}
// NOLINTEND(bugprone-unchecked-optional-access)

TEST_CASE("dynamic pass graph accepts move-only steps", "[vkexec][pass]")
{
  constexpr int k_initial_value = 42;
  vkexec::scheduler sched{ nullptr };
  auto graph = vkexec::make_dynamic_pass_graph(ex::schedule(sched));
  graph.append(vkexec::make_pass_adaptor(move_only_pass_step{ k_initial_value }));
  REQUIRE(graph.steps.size() == 1);
}

TEST_CASE("copyable binding graphs defer mutable state to each operation", "[vkexec][pass]")
{
  vkexec::handles::compute_pipeline const pipe{};
  auto graph = ex::transform_sender(
    ex::schedule(vkexec::scheduler{ nullptr }) | vkexec::bind_resources(pipe, vkexec::resource_table{}), ex::env<>{});

  std::promise<void> first_completed;
  std::promise<void> second_completed;
  auto first = graph.connect(completion_probe_receiver{ .completed = &first_completed });
  auto second = graph.connect(completion_probe_receiver{ .completed = &second_completed });

  REQUIRE(std::get<0>(graph.steps).state == nullptr);
  REQUIRE(std::get<0>(first.steps).state == nullptr);
  REQUIRE(std::get<0>(second.steps).state == nullptr);
}

TEST_CASE("static pass recording stops at the first failure", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  int first_count = 0;
  int failing_count = 0;
  int skipped_count = 0;
  std::tuple steps{
    counting_pass_step{ .record_count = &first_count, .should_fail = false },
    counting_pass_step{ .record_count = &failing_count, .should_fail = true },
    counting_pass_step{ .record_count = &skipped_count, .should_fail = false },
  };
  vkexec::detail::pass_cleanup cleanup{};

  auto recorded = vkexec::detail::record_static_steps(*ctx, VK_NULL_HANDLE, cleanup, steps);

  REQUIRE_FALSE(recorded);
  REQUIRE(first_count == 1);
  REQUIRE(failing_count == 1);
  REQUIRE(skipped_count == 0);
}

TEST_CASE("dynamic pass recording stops at first failure", "[vkexec][pass][gpu]")
{
  auto ctx = vkexec::test::require_context();
  int first_count = 0;
  int failing_count = 0;
  int skipped_count = 0;

  auto graph = vkexec::make_dynamic_pass_graph(ex::schedule(ctx->get_scheduler()));
  graph.append(vkexec::make_pass_adaptor(counting_pass_step{ .record_count = &first_count, .should_fail = false }));
  graph.append(vkexec::make_pass_adaptor(counting_pass_step{ .record_count = &failing_count, .should_fail = true }));
  graph.append(vkexec::make_pass_adaptor(counting_pass_step{ .record_count = &skipped_count, .should_fail = false }));

  vkexec::detail::pass_cleanup cleanup{};
  auto recorded = vkexec::detail::record_dynamic_steps(*ctx, VK_NULL_HANDLE, cleanup, graph.steps);

  REQUIRE_FALSE(recorded);
  REQUIRE(first_count == 1);
  REQUIRE(failing_count == 1);
  REQUIRE(skipped_count == 0);
}
