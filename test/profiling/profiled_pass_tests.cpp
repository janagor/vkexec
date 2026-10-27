#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier_params.hpp>
#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/resource_state_tracker.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/result.hpp>
#include <vkexec/submit_scope.hpp>
#include <vkexec_profiling/profile_event.hpp>
#include <vkexec_profiling/profiled.hpp>
#include <vkexec_profiling/profiler.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <future>
#if VKEXEC_HAS_EXCEPTIONS
#include <stdexcept>
#endif
#include <type_traits>
#include <utility>

namespace {

struct mutable_declaration_step
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  int *declarations{ nullptr };

  [[nodiscard]] static auto has_resource_declarations() noexcept -> bool { return true; }
  auto declare_resources(vkexec::detail::resource_state_tracker &tracker,
    std::size_t index,
    vkexec::queue_ref queue) const -> vkexec::status
  {
    ++*declarations;
    constexpr VkDeviceSize k_buffer_size = 64;
    return tracker.use(
      vkexec::write(buffer, 0, k_buffer_size, vkexec::buffer_usage::storage_compute), index, queue.family, queue.queue);
  }
  static auto record(vkexec::context & /*ctx*/, VkCommandBuffer /*cmd*/, vkexec::detail::pass_cleanup & /*cleanup*/)
    -> vkexec::status
  { return {}; }
};

struct four_argument_declaration_step
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  bool write{ false };
  bool *crosses_families_seen{ nullptr };

  [[nodiscard]] static auto has_resource_declarations() noexcept -> bool { return true; }
  auto declare_resources(vkexec::detail::resource_state_tracker &tracker,
    std::size_t index,
    vkexec::queue_ref queue,
    bool crosses_families) const -> vkexec::status
  {
    *crosses_families_seen = crosses_families;
    constexpr VkDeviceSize k_buffer_size = 64;
    if (write) {
      return tracker.use(vkexec::write(buffer, 0, k_buffer_size, vkexec::buffer_usage::storage_compute),
        index,
        queue.family,
        queue.queue);
    }
    return tracker.use(
      vkexec::read(buffer, 0, k_buffer_size, vkexec::buffer_usage::storage_compute), index, queue.family, queue.queue);
  }
  static auto record(vkexec::context & /*ctx*/, VkCommandBuffer /*cmd*/, vkexec::detail::pass_cleanup & /*cleanup*/)
    -> vkexec::status
  { return {}; }
};

struct nonassignable_step
{
  nonassignable_step() = default;
  nonassignable_step(nonassignable_step const &) = delete;
  auto operator=(nonassignable_step const &) -> nonassignable_step & = delete;
  nonassignable_step(nonassignable_step &&) = default;
  auto operator=(nonassignable_step &&) -> nonassignable_step & = delete;
  ~nonassignable_step() = default;
  static auto record(vkexec::context & /*ctx*/, VkCommandBuffer /*cmd*/, vkexec::detail::pass_cleanup & /*cleanup*/)
    -> vkexec::status
  { return {}; }
};

struct unrelated_publish_step
{
  bool *called{ nullptr };
  static auto record(vkexec::context & /*ctx*/, VkCommandBuffer /*cmd*/, vkexec::detail::pass_cleanup & /*cleanup*/)
    -> vkexec::status
  { return {}; }
  auto publish_after_gpu() const -> void { *called = true; }
};

struct completion_receiver
{
  using receiver_concept = stdexec::receiver_t;
  struct env
  {
  };
  std::promise<void> *done{ nullptr };

  auto set_value() const && noexcept -> void { done->set_value(); }
  auto set_error(vkexec::error const & /*error*/) const && noexcept -> void { done->set_value(); }
  auto set_stopped() const && noexcept -> void { done->set_value(); }
  [[nodiscard]] static auto get_env() noexcept -> env { return {}; }
};

static_assert(vkexec::detail::static_pass_step<nonassignable_step>);
static_assert(std::move_constructible<vkexec::profiling::detail::profiled_pass_step<nonassignable_step>>);
static_assert(!std::is_move_assignable_v<vkexec::profiling::detail::profiled_pass_step<nonassignable_step>>);

}// namespace

TEST_CASE("unprofiled steps do not opt into profiling publication", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  bool called = false;
  auto graph =
    stdexec::schedule(ctx->get_scheduler()) | vkexec::make_pass_adaptor(unrelated_publish_step{ .called = &called });
  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
  REQUIRE_FALSE(called);
}

TEST_CASE("profiled four-argument declarations survive dynamic type erasure", "[vkexec][profiling]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx);
  REQUIRE(profiler);
  char storage{};
  auto *const buffer = reinterpret_cast<VkBuffer>(&storage);// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
  constexpr std::size_t k_step_count = 2;
  constexpr std::size_t k_writer_index = 0;
  constexpr std::size_t k_reader_index = 1;
  constexpr std::uint32_t k_writer_family = 0;
  constexpr std::uint32_t k_reader_family = 1;
  auto const writer_queue = vkexec::queue_ref{ .family = k_writer_family };
  auto const reader_queue = vkexec::queue_ref{ .family = k_reader_family };
  bool static_crosses = false;
  bool dynamic_crosses = false;
  using step_type = vkexec::profiling::detail::profiled_pass_step<four_argument_declaration_step>;
  step_type static_writer{
    { .buffer = buffer, .write = true, .crosses_families_seen = &static_crosses }, "writer", *profiler
  };
  step_type static_reader{
    { .buffer = buffer, .write = false, .crosses_families_seen = &static_crosses }, "reader", *profiler
  };
  vkexec::detail::resource_state_tracker static_tracker{ k_step_count };
  REQUIRE(vkexec::detail::declare_step_resources(static_writer, static_tracker, k_writer_index, writer_queue, true));
  REQUIRE(vkexec::detail::declare_step_resources(static_reader, static_tracker, k_reader_index, reader_queue, true));
  auto const static_plan = std::move(static_tracker).finish();

  vkexec::detail::dynamic_pass_step dynamic_writer{ step_type{
    { .buffer = buffer, .write = true, .crosses_families_seen = &dynamic_crosses }, "writer", *profiler } };
  vkexec::detail::dynamic_pass_step dynamic_reader{ step_type{
    { .buffer = buffer, .write = false, .crosses_families_seen = &dynamic_crosses }, "reader", *profiler } };
  vkexec::detail::resource_state_tracker dynamic_tracker{ k_step_count };
  REQUIRE(vkexec::detail::declare_step_resources(dynamic_writer, dynamic_tracker, k_writer_index, writer_queue, true));
  REQUIRE(vkexec::detail::declare_step_resources(dynamic_reader, dynamic_tracker, k_reader_index, reader_queue, true));
  auto const dynamic_plan = std::move(dynamic_tracker).finish();
  REQUIRE(static_crosses);
  REQUIRE(dynamic_crosses);
  REQUIRE(static_plan.dependencies == dynamic_plan.dependencies);
  REQUIRE_FALSE(static_plan.steps.at(k_writer_index).buffers_after.empty());
  REQUIRE_FALSE(dynamic_plan.steps.at(k_writer_index).buffers_after.empty());
  REQUIRE_FALSE(static_plan.steps.at(k_reader_index).buffers_before.empty());
  REQUIRE_FALSE(dynamic_plan.steps.at(k_reader_index).buffers_before.empty());
}

TEST_CASE("profiled primitive passes publish after static graph completion", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx);
  REQUIRE(profiler);
  if (!profiler->supported(ctx->compute_queue_ref())) { SKIP("compute queue has no timestamp support"); }

  auto pass = []() -> auto { return vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {}); };
  auto graph = stdexec::schedule(ctx->get_scheduler()) | vkexec::profiling::profiled(*profiler, "first", pass())
               | vkexec::profiling::profiled(*profiler, "second", pass());
  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
  auto events = profiler->take_completed();
  constexpr std::size_t k_region_count = 2;
  REQUIRE(events.size() == k_region_count);
  REQUIRE(events.at(0).name == "first");
  REQUIRE(events.at(1).name == "second");
  REQUIRE(events.at(0).submission_id == events.at(1).submission_id);
  REQUIRE(events.at(0).status == vkexec::profiling::sample_status::valid);
  REQUIRE(events.at(1).status == vkexec::profiling::sample_status::valid);
}

TEST_CASE("profiled primitive passes work in dynamic graphs", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx);
  REQUIRE(profiler);
  if (!profiler->supported(ctx->compute_queue_ref())) { SKIP("compute queue has no timestamp support"); }

  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  graph.append(vkexec::profiling::profiled(
    *profiler, "dynamic", vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {})));
  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
  auto events = profiler->take_completed();
  constexpr std::size_t k_region_count = 1;
  REQUIRE(events.size() == k_region_count);
  REQUIRE(events.front().name == "dynamic");
  REQUIRE(events.front().queue_family == ctx->compute_queue_ref().family);
  REQUIRE(events.front().status == vkexec::profiling::sample_status::valid);
}

TEST_CASE("profiled graph rejects a profiler from another device", "[vkexec][profiling][gpu]")
{
  auto profiler_context = vkexec::test::require_context();
  auto graph_context = vkexec::test::require_context();
  REQUIRE(profiler_context->device() != graph_context->device());
  auto profiler = vkexec::profiling::profiler::create(*profiler_context);
  REQUIRE(profiler);
  int recorded = 0;
  auto graph = stdexec::schedule(graph_context->get_scheduler())
               | vkexec::profiling::profiled(*profiler,
                 "foreign-device",
                 vkexec::custom_pass(vkexec::uses(), [&recorded](VkCommandBuffer /*cmd*/) -> void { ++recorded; }));
  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(outcome.failed());
  REQUIRE(outcome.error.has_value());
  REQUIRE(outcome.error.value().code == vkexec::make_error_code(vkexec::errc::invalid_argument));
  REQUIRE(recorded == 0);
  REQUIRE(profiler->take_completed().empty());
}

TEST_CASE("profiling keeps resource hazards visible to graph planning", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx);
  REQUIRE(profiler);
  char storage{};
  auto *const buffer = reinterpret_cast<VkBuffer>(&storage);// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
  constexpr VkDeviceSize k_buffer_size = 64;
  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  graph.append(vkexec::profiling::profiled(*profiler,
    "write",
    vkexec::custom_pass(vkexec::uses(vkexec::write(buffer, 0, k_buffer_size, vkexec::buffer_usage::storage_compute)),
      [](VkCommandBuffer /*cmd*/) -> void {})));
  graph.append(vkexec::profiling::profiled(*profiler,
    "read",
    vkexec::custom_pass(vkexec::uses(vkexec::read(buffer, 0, k_buffer_size, vkexec::buffer_usage::storage_compute)),
      [](VkCommandBuffer /*cmd*/) -> void {})));
  auto execution = vkexec::detail::plan_execution(graph.steps.size(), graph.step_queues, ctx->compute_queue_ref());
  REQUIRE(execution);
  auto synchronization = vkexec::detail::plan_resource_sync(*execution, graph.steps);
  REQUIRE(synchronization);
  constexpr std::size_t k_read_step = 1;
  REQUIRE_FALSE(synchronization->steps.at(k_read_step).buffers_before.empty());
}

TEST_CASE("mutable-only resource declarations survive profiling", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx);
  REQUIRE(profiler);
  char storage{};
  auto *const buffer = reinterpret_cast<VkBuffer>(&storage);// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
  int declarations = 0;
  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  graph.append(vkexec::profiling::profiled(*profiler,
    "mutable",
    vkexec::make_pass_adaptor(mutable_declaration_step{ .buffer = buffer, .declarations = &declarations })));
  auto execution = vkexec::detail::plan_execution(graph.steps.size(), graph.step_queues, ctx->compute_queue_ref());
  REQUIRE(execution);
  auto synchronization = vkexec::detail::plan_resource_sync(*execution, graph.steps);
  REQUIRE(synchronization);
  REQUIRE(declarations == 1);
}

TEST_CASE("cross-family synchronization plan is unchanged by profiling", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto const compute = ctx->compute_queue_ref();
  auto const graphics = ctx->graphics_queue_ref();
  if (graphics.queue == VK_NULL_HANDLE || compute.family == graphics.family) {
    SKIP("distinct queue families are unavailable");
  }
  auto profiler = vkexec::profiling::profiler::create(*ctx);
  REQUIRE(profiler);
  char storage{};
  auto *const buffer = reinterpret_cast<VkBuffer>(&storage);// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
  constexpr VkDeviceSize k_buffer_size = 64;
  auto make_write = [&]() -> auto {
    return vkexec::custom_pass(
      vkexec::uses(vkexec::write(buffer, 0, k_buffer_size, vkexec::buffer_usage::storage_compute)),
      [](VkCommandBuffer /*cmd*/) -> void {});
  };
  auto make_read = [&]() -> auto {
    return vkexec::custom_pass(
      vkexec::uses(vkexec::read(buffer, 0, k_buffer_size, vkexec::buffer_usage::storage_compute)),
      [](VkCommandBuffer /*cmd*/) -> void {});
  };
  auto plain = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  plain.append_queue(compute);
  plain.append(make_write());
  plain.append_queue(graphics);
  plain.append(make_read());
  auto measured = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  measured.append_queue(compute);
  measured.append(vkexec::profiling::profiled(*profiler, "write", make_write()));
  measured.append_queue(graphics);
  measured.append(vkexec::profiling::profiled(*profiler, "read", make_read()));
  auto plain_execution = vkexec::detail::plan_execution(plain.steps.size(), plain.step_queues, compute);
  auto measured_execution = vkexec::detail::plan_execution(measured.steps.size(), measured.step_queues, compute);
  REQUIRE(plain_execution);
  REQUIRE(measured_execution);
  auto plain_sync = vkexec::detail::plan_resource_sync(*plain_execution, plain.steps);
  auto measured_sync = vkexec::detail::plan_resource_sync(*measured_execution, measured.steps);
  REQUIRE(plain_sync);
  REQUIRE(measured_sync);
  constexpr std::size_t k_write_step = 0;
  constexpr std::size_t k_read_step = 1;
  REQUIRE(plain_sync->dependencies == measured_sync->dependencies);
  REQUIRE(plain_sync->steps.at(k_write_step).buffers_after.size()
          == measured_sync->steps.at(k_write_step).buffers_after.size());
  REQUIRE(plain_sync->steps.at(k_read_step).buffers_before.size()
          == measured_sync->steps.at(k_read_step).buffers_before.size());
  REQUIRE_FALSE(plain_sync->steps.at(k_write_step).buffers_after.empty());
  REQUIRE_FALSE(plain_sync->steps.at(k_read_step).buffers_before.empty());
  auto same_barrier = [](
                        vkexec::buffer_barrier_params const &left, vkexec::buffer_barrier_params const &right) -> bool {
    return left.buffer == right.buffer && left.offset == right.offset && left.size == right.size
           && left.src_stage == right.src_stage && left.dst_stage == right.dst_stage
           && left.src_access == right.src_access && left.dst_access == right.dst_access
           && left.src_queue_family == right.src_queue_family && left.dst_queue_family == right.dst_queue_family;
  };
  REQUIRE(std::ranges::equal(plain_sync->steps.at(k_write_step).buffers_after,
    measured_sync->steps.at(k_write_step).buffers_after,
    same_barrier));
  REQUIRE(std::ranges::equal(plain_sync->steps.at(k_read_step).buffers_before,
    measured_sync->steps.at(k_read_step).buffers_before,
    same_barrier));
}

TEST_CASE("copying a profiled graph gives each execution its own queries", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx);
  REQUIRE(profiler);
  if (!profiler->supported(ctx->compute_queue_ref())) { SKIP("compute queue has no timestamp support"); }
  auto graph = stdexec::schedule(ctx->get_scheduler())
               | vkexec::profiling::profiled(
                 *profiler, "copy", vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {}));
  static_assert(std::copy_constructible<decltype(graph)>);
  auto second = graph;
  auto first_result = vkexec::test::sync_wait_sender(std::move(graph));
  auto second_result = vkexec::test::sync_wait_sender(std::move(second));
  REQUIRE(vkexec::test::sync_wait_completed(first_result));
  REQUIRE(vkexec::test::sync_wait_completed(second_result));
  auto events = profiler->take_completed();
  constexpr std::size_t k_execution_count = 2;
  REQUIRE(events.size() == k_execution_count);
  REQUIRE(events.at(0).submission_id != events.at(1).submission_id);
}

TEST_CASE("two profiled graph operations complete independently", "[vkexec][profiling][gpu]")
{
  constexpr std::uint32_t k_queries_per_pool = 4;
  constexpr std::uint32_t k_max_in_flight_regions = 2;
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(
    *ctx, { .queries_per_pool = k_queries_per_pool, .max_in_flight_regions = k_max_in_flight_regions });
  REQUIRE(profiler);
  if (!profiler->supported(ctx->compute_queue_ref())) { SKIP("compute queue has no timestamp support"); }
  auto pass = []() -> auto { return vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {}); };
  auto first = stdexec::schedule(ctx->get_scheduler()) | vkexec::profiling::profiled(*profiler, "first", pass());
  auto second = stdexec::schedule(ctx->get_scheduler()) | vkexec::profiling::profiled(*profiler, "second", pass());
  std::promise<void> first_done;
  std::promise<void> second_done;
  auto first_finished = first_done.get_future();
  auto second_finished = second_done.get_future();
  auto first_operation = stdexec::connect(std::move(first), completion_receiver{ .done = &first_done });
  auto second_operation = stdexec::connect(std::move(second), completion_receiver{ .done = &second_done });
  stdexec::start(first_operation);
  stdexec::start(second_operation);
  constexpr auto k_completion_timeout = std::chrono::seconds{ 5 };
  auto const first_status = first_finished.wait_for(k_completion_timeout);
  auto const second_status = second_finished.wait_for(k_completion_timeout);
  REQUIRE(first_status == std::future_status::ready);
  REQUIRE(second_status == std::future_status::ready);
  auto events = profiler->take_completed();
  constexpr std::size_t k_execution_count = 2;
  REQUIRE(events.size() == k_execution_count);
  REQUIRE(events.at(0).submission_id != events.at(1).submission_id);
  REQUIRE(events.at(0).status == vkexec::profiling::sample_status::valid);
  REQUIRE(events.at(1).status == vkexec::profiling::sample_status::valid);
}

TEST_CASE("query capacity errors release unsubmitted graph allocations", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx, { .max_in_flight_regions = 1 });
  REQUIRE(profiler);
  if (!profiler->supported(ctx->compute_queue_ref())) { SKIP("compute queue has no timestamp support"); }
  auto pass = []() -> auto { return vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {}); };
  auto failed_graph = stdexec::schedule(ctx->get_scheduler()) | vkexec::profiling::profiled(*profiler, "first", pass())
                      | vkexec::profiling::profiled(*profiler, "second", pass());
  auto failed = vkexec::test::sync_wait_sender(std::move(failed_graph));
  REQUIRE(failed.failed());
  REQUIRE(profiler->take_completed().empty());

  auto retry = stdexec::schedule(ctx->get_scheduler()) | vkexec::profiling::profiled(*profiler, "retry", pass());
  auto completed = vkexec::test::sync_wait_sender(std::move(retry));
  REQUIRE(vkexec::test::sync_wait_completed(completed));
  REQUIRE(profiler->take_completed().size() == 1);
}

TEST_CASE("recording failure releases a profiled query pair", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx, { .max_in_flight_regions = 1 });
  REQUIRE(profiler);
  if (!profiler->supported(ctx->compute_queue_ref())) { SKIP("compute queue has no timestamp support"); }
  auto failing = vkexec::detail::make_callback_pass_step(
    [](vkexec::context &, VkCommandBuffer, vkexec::detail::pass_cleanup &) -> vkexec::status {
      return vkexec::fail(vkexec::errc::invalid_argument, "injected recording failure");
    });
  auto graph = stdexec::schedule(ctx->get_scheduler())
               | vkexec::profiling::profiled(*profiler, "failed", vkexec::make_pass_adaptor(failing));
  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(outcome.failed());
  REQUIRE(profiler->take_completed().empty());

  auto retry = stdexec::schedule(ctx->get_scheduler())
               | vkexec::profiling::profiled(
                 *profiler, "retry", vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {}));
  auto completed = vkexec::test::sync_wait_sender(std::move(retry));
  REQUIRE(vkexec::test::sync_wait_completed(completed));
  REQUIRE(profiler->take_completed().size() == 1);
}

TEST_CASE("profiled passes keep missing metadata errors", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx);
  REQUIRE(profiler);
  auto graph = stdexec::schedule(ctx->get_scheduler())
               | vkexec::profiling::profiled(
                 *profiler, "missing", vkexec::compute_pass(vkexec::compute_bind{}, vkexec::dispatch{}));
  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(outcome.failed());
  REQUIRE(outcome.error.has_value());
  REQUIRE(outcome.error.value().code == vkexec::make_error_code(vkexec::errc::invalid_argument));
  REQUIRE(profiler->take_completed().empty());
}

TEST_CASE("profiled DAG branches retain their execution queue", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx);
  REQUIRE(profiler);
  auto const compute = ctx->compute_queue_ref();
  auto const graphics = ctx->graphics_queue_ref();
  if (!profiler->supported(compute) || !profiler->supported(graphics)
      || vkexec::detail::same_queue(compute, graphics)) {
    SKIP("distinct timestamp-capable queues are unavailable");
  }
  auto pass = []() -> auto { return vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {}); };
  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  graph.append_queue(compute);
  auto const compute_index = graph.append_after({}, vkexec::profiling::profiled(*profiler, "compute", pass()));
  graph.append_queue(graphics);
  auto const graphics_index = graph.append_after({}, vkexec::profiling::profiled(*profiler, "graphics", pass()));
  REQUIRE(compute_index != graphics_index);
  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
  auto events = profiler->take_completed();
  constexpr std::size_t k_branch_count = 2;
  REQUIRE(events.size() == k_branch_count);
  for (auto const &event : events) {
    REQUIRE(event.status == vkexec::profiling::sample_status::valid);
    if (event.name == "compute") {
      REQUIRE(event.queue_family == compute.family);
    } else if (event.name == "graphics") {
      REQUIRE(event.queue_family == graphics.family);
    } else {
      FAIL("unexpected profiled branch");
    }
  }
}

TEST_CASE("original after-GPU callback runs before publishing", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx);
  REQUIRE(profiler);
  if (!profiler->supported(ctx->compute_queue_ref())) { SKIP("compute queue has no timestamp support"); }
  bool callback_ran = false;
  auto step = vkexec::detail::make_callback_pass_step(
    [](vkexec::context &, VkCommandBuffer, vkexec::detail::pass_cleanup &) -> vkexec::status { return {}; },
    [&callback_ran]() -> void { callback_ran = true; });
  auto graph = stdexec::schedule(ctx->get_scheduler())
               | vkexec::profiling::profiled(*profiler, "callback", vkexec::make_pass_adaptor(step));
  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
  REQUIRE(callback_ran);
  REQUIRE(profiler->take_completed().size() == 1);
}

#if VKEXEC_HAS_EXCEPTIONS
TEST_CASE("later after-GPU callback failure suppresses completed samples", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx);
  REQUIRE(profiler);
  if (!profiler->supported(ctx->compute_queue_ref())) { SKIP("compute queue has no timestamp support"); }
  auto record = [](vkexec::context &, VkCommandBuffer, vkexec::detail::pass_cleanup &) -> vkexec::status { return {}; };
  auto first = vkexec::detail::make_callback_pass_step(record, []() -> void {});
  auto second =
    vkexec::detail::make_callback_pass_step(record, []() -> void { throw std::runtime_error("after_gpu failed"); });
  auto graph = stdexec::schedule(ctx->get_scheduler())
               | vkexec::profiling::profiled(*profiler, "first", vkexec::make_pass_adaptor(first))
               | vkexec::profiling::profiled(*profiler, "second", vkexec::make_pass_adaptor(second));
  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(outcome.failed());
  REQUIRE(profiler->take_completed().empty());
}
#endif
