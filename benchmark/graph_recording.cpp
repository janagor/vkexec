#include "../test/support/recording_buffer.hpp"
#include "../test/support/recording_compute.hpp"
#include "recording_allocations.hpp"

#include <vkexec/barrier.hpp>
#include <vkexec/barrier_params.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/pass_execution.hpp>
#include <vkexec/detail/pass_steps.hpp>
#include <vkexec/detail/pass_types.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/resource_use.hpp>
#include <stdexec/execution.hpp>
#include <vkexec/error.hpp>
#include <vkexec/record_into.hpp>
#include <vkexec/recorded_graph.hpp>
#include <vkexec/result.hpp>
#include <vkexec/submit_scope.hpp>
#include <vkexec/sync_wait.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace {
constexpr std::uint32_t k_query_count = 2;
constexpr std::uint32_t k_start_query = 0;
constexpr std::uint32_t k_end_query = 1;
constexpr std::uint32_t k_dispatch_extent = 1;
constexpr std::uint32_t k_no_dynamic_offsets = 0;

struct timestamp_queries
{
  VkDevice device{};
  VkQueryPool pool{};
  explicit timestamp_queries(VkDevice value) : device(value) {}
  timestamp_queries(timestamp_queries const &) = delete;
  auto operator=(timestamp_queries const &) -> timestamp_queries & = delete;
  timestamp_queries(timestamp_queries &&) = delete;
  auto operator=(timestamp_queries &&) -> timestamp_queries & = delete;
  ~timestamp_queries()
  {
    if (pool != VK_NULL_HANDLE) { vkDestroyQueryPool(device, pool, nullptr); }
  }
};
struct compute_step
{
  decltype(vkexec::uses(vkexec::buffer_use{})) resources;
  vkexec::compute_bind bind;
  VkQueryPool queries{};
  bool first{};
  bool last{};
  auto record(vkexec::context &ctx, VkCommandBuffer cmd, vkexec::detail::pass_cleanup &cleanup) const -> vkexec::status
  {
    (void)ctx;
    (void)cleanup;
    if (queries != VK_NULL_HANDLE && first) {
      vkCmdResetQueryPool(cmd, queries, k_start_query, k_query_count);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, k_start_query);
    }
    // record_pass returns void, matching vkCmdDispatch; it has no recording status.
    vkexec::record_pass(cmd,
      bind,
      nullptr,
      k_no_dynamic_offsets,
      vkexec::dispatch{ .x = k_dispatch_extent, .y = k_dispatch_extent, .z = k_dispatch_extent });
    if (queries != VK_NULL_HANDLE && last) { vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, k_end_query); }
    return {};
  }
};

auto make_graph(vkexec::context &ctx,
  VkBuffer buffer,
  vkexec::compute_bind const &bind,
  std::size_t count,
  VkQueryPool queries = VK_NULL_HANDLE,
  bool dag = false) -> vkexec::dynamic_pass_graph_sender
{
  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx.get_scheduler()));
  for (std::size_t index{}; index < count; ++index) {
    auto pass = vkexec::make_pass_adaptor(
      compute_step{ .resources = vkexec::uses(vkexec::read_write(
                      buffer, 0, vkexec::test::recording_buffer::k_byte_size, vkexec::buffer_usage::storage_compute)),
        .bind = bind,
        .queries = queries,
        .first = index == 0,
        .last = index + 1 == count });
    if (dag) {
      std::vector<std::size_t> incoming;
      if (index != 0) { incoming.push_back(index - 1); }
      (void)graph.append_after(incoming, std::move(pass));
    } else {
      graph.append(std::move(pass));
    }
  }
  return graph;
}

enum class mode : std::uint8_t { manual, owned, external_prepared, external, planning, owned_execution, owned_dag_execution };

struct recording_metrics
{
  double gpu_ns{};
  std::size_t allocations{};
  std::size_t submitted_scopes{};
};

struct recording_fixture
{
  vkexec::context *ctx;
  VkBuffer buffer;
  vkexec::compute_bind const *bind;
  std::size_t count;
  vkexec::detail::prepared_graph const *prepared;
  vkexec::buffer_barrier_params const *barrier;
  timestamp_queries const *queries;
  std::uint32_t timestamp_bits;
  float timestamp_period;
  std::size_t scopes_per_execution;
};

auto timestamp_duration(recording_fixture const &fixture) -> double
{
  if (fixture.queries->pool == VK_NULL_HANDLE) { return {}; }
  std::array<std::uint64_t, k_query_count> ticks{};
  if (vkGetQueryPoolResults(fixture.ctx->device(),
        fixture.queries->pool,
        k_start_query,
        k_query_count,
        sizeof(ticks),
        ticks.data(),
        sizeof(ticks.front()),
        VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) {
    return {};
  }
  constexpr auto k_timestamp_width = std::numeric_limits<std::uint64_t>::digits;
  auto const mask = fixture.timestamp_bits == k_timestamp_width
                      ? ~std::uint64_t{}
                      : (std::uint64_t{ 1 } << fixture.timestamp_bits) - 1;
  return static_cast<double>((ticks.back() - ticks.front()) & mask) * static_cast<double>(fixture.timestamp_period);
}

template<mode Mode> auto record_commands(
  recording_fixture const &fixture,
  vkexec::detail::submit_scope &scope,
  vkexec::dynamic_pass_graph_sender &input,
  vkexec::result<vkexec::recorded_graph> &recorded) -> vkexec::status
{
  if constexpr (Mode == mode::manual) {
    for (std::size_t index{}; index < fixture.count; ++index) {
      if (index != 0) {
        auto const status = vkexec::buffer_barrier(*fixture.ctx, scope.cmd, *fixture.barrier);
        if (!status) { return status; }
      }
      vkexec::record_pass(scope.cmd,
        *fixture.bind,
        nullptr,
        k_no_dynamic_offsets,
        vkexec::dispatch{ .x = k_dispatch_extent, .y = k_dispatch_extent, .z = k_dispatch_extent });
    }
    return {};
  } else if constexpr (Mode == mode::owned) {
    auto const info = vkexec::detail::recording_info(scope.queue);
    for (auto const &batch : fixture.prepared->execution.batches) {
      auto const status = vkexec::detail::record_sync_batch(*fixture.ctx,
        scope.cmd, scope.cleanup, info, batch, fixture.prepared->synchronization, input.steps);
      if (!status) { return status; }
    }
    return {};
  } else if constexpr (Mode == mode::external_prepared) {
    return vkexec::detail::record_external(
      *fixture.ctx, scope.cmd, scope.queue, *fixture.prepared, input.steps, scope.cleanup);
  } else {
    recorded = vkexec::record_into(*fixture.ctx, scope.cmd, std::move(input), { scope.queue });
    return {};
  }
}

auto discard_recording(vkexec::detail::submit_scope &scope,
  vkexec::result<vkexec::recorded_graph> &recorded) -> void
{
  // Reset commands before releasing resources retained by an unsubmitted recording.
  if (vkResetCommandBuffer(scope.cmd, 0) != VK_SUCCESS) { std::terminate(); }
  if (recorded) {
    auto const abandoned = recorded->abandon();
    if (!abandoned) { std::terminate(); }
  }
}

auto submit_recording(benchmark::State &state,
  recording_fixture const &fixture,
  vkexec::detail::submit_scope &scope,
  vkexec::result<vkexec::recorded_graph> &recorded,
  recording_metrics &metrics) -> void
{
  auto const submitted = fixture.ctx->submit_and_wait(scope.cmd, scope.queue);
  if (submitted) { metrics.gpu_ns += timestamp_duration(fixture); }
  if (recorded) {
    // Submission may fail after the queue accepted the buffer. Its resources
    // cannot be abandoned while that work may still be in flight.
    if (!submitted) { std::terminate(); }
    auto const marked = recorded->mark_submitted();
    auto const completed = marked ? recorded->complete() : marked;
    if (!completed) { state.SkipWithError(completed.error().message()); }
  }
  if (!submitted) { state.SkipWithError("submission failed"); }
  scope.release();
}

template<mode Mode> auto record_iteration(benchmark::State &state,
  recording_fixture const &fixture,
  vkexec::dynamic_pass_graph_sender &input,
  recording_metrics &metrics) -> bool
{
  auto opened = vkexec::detail::submit_scope::open(*fixture.ctx);
  if (!opened) {
    state.SkipWithError(opened.error().message());
    state.ResumeTiming();
    return false;
  }
  auto scope = vkexec::expected_take(opened);
  vkexec::result<vkexec::recorded_graph> recorded = vkexec::fail(vkexec::errc::empty_result);
  if (fixture.queries->pool != VK_NULL_HANDLE) {
    vkCmdResetQueryPool(scope.cmd, fixture.queries->pool, k_start_query, k_query_count);
    vkCmdWriteTimestamp(scope.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, fixture.queries->pool, k_start_query);
  }
  state.ResumeTiming();
  recording_allocations::begin();
  auto const status = record_commands<Mode>(fixture, scope, input, recorded);
  metrics.allocations += recording_allocations::end();
  state.PauseTiming();
  if (fixture.queries->pool != VK_NULL_HANDLE) {
    vkCmdWriteTimestamp(scope.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, fixture.queries->pool, k_end_query);
  }
  if (!status || (Mode == mode::external && !recorded)) {
    discard_recording(scope, recorded);
    state.SkipWithError("recording failed");
    state.ResumeTiming();
    return false;
  }
  auto const ended = scope.end_recording();
  if (!ended) {
    discard_recording(scope, recorded);
    state.SkipWithError(ended.error().message());
    state.ResumeTiming();
    return false;
  }
  submit_recording(state, fixture, scope, recorded, metrics);
  state.ResumeTiming();
  return true;
}

auto execute_iteration(benchmark::State &state,
  recording_fixture const &fixture,
  vkexec::dynamic_pass_graph_sender &input,
  recording_metrics &metrics) -> bool
{
  state.ResumeTiming();
  recording_allocations::begin();
  auto done = vkexec::try_sync_wait(std::move(input));
  metrics.allocations += recording_allocations::end();
  if (done.failed()) {
    state.SkipWithError("owned execution failed");
    return false;
  }
  state.PauseTiming();
  metrics.submitted_scopes += fixture.scopes_per_execution;
  metrics.gpu_ns += timestamp_duration(fixture);
  state.ResumeTiming();
  return true;
}

template<mode Mode> auto run_iterations(benchmark::State &state, recording_fixture const &fixture) -> recording_metrics
{
  recording_metrics metrics;
  constexpr bool k_executes_owned = Mode == mode::owned_execution || Mode == mode::owned_dag_execution;
  for (auto const iteration : state) {
    (void)iteration;
    state.PauseTiming();
    auto input = make_graph(*fixture.ctx, fixture.buffer, *fixture.bind, fixture.count,
      k_executes_owned ? fixture.queries->pool : VK_NULL_HANDLE, Mode == mode::owned_dag_execution);
    if constexpr (Mode == mode::planning) {
      state.ResumeTiming();
      recording_allocations::begin();
      auto plan = vkexec::detail::prepare_graph(
        *fixture.ctx, input.steps, input.step_queues, input.step_predecessors, fixture.ctx->compute_queue_ref());
      metrics.allocations += recording_allocations::end();
      benchmark::DoNotOptimize(plan);
    } else if constexpr (k_executes_owned) {
      if (!execute_iteration(state, fixture, input, metrics)) { break; }
    } else {
      if (!record_iteration<Mode>(state, fixture, input, metrics)) { break; }
    }
    state.PauseTiming();
    input = vkexec::dynamic_pass_graph_sender{};
    state.ResumeTiming();
  }
  return metrics;
}

template<mode Mode> auto recording(benchmark::State &state) -> void
{
  auto made = vkexec::try_sync_wait_value(vkexec::factory::make_context());
  if (!made) {
    state.SkipWithError(made.error().message());
    return;
  }
  auto ctx = std::move(*made);
  auto allocated = vkexec::test::recording_buffer::make(*ctx);
  if (!allocated) {
    state.SkipWithError(allocated.error().message());
    return;
  }
  auto buffer = std::move(*allocated);
  auto const count = static_cast<std::size_t>(state.range(0));
  auto compute_result = vkexec::test::recording_compute::make(*ctx, buffer->buffer, buffer->k_byte_size);
  if (!compute_result) {
    state.SkipWithError(compute_result.error().message());
    return;
  }
  auto compute = std::move(*compute_result);
  constexpr bool k_dag = Mode == mode::owned_dag_execution;
  auto graph = make_graph(*ctx, buffer->buffer, compute->bind, count, VK_NULL_HANDLE, k_dag);
  auto prepared = vkexec::detail::prepare_graph(
    *ctx, graph.steps, graph.step_queues, graph.step_predecessors, ctx->compute_queue_ref());
  if (!prepared) {
    state.SkipWithError(prepared.error().message());
    return;
  }
  std::size_t barrier_count{};
  for (auto const &point : prepared->synchronization.steps) {
    barrier_count +=
      point.images_before.size() + point.images_after.size() + point.buffers_before.size() + point.buffers_after.size();
  }
  vkexec::buffer_barrier_params const barrier{ .buffer = buffer->buffer,
    .size = buffer->k_byte_size,
    .src_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .src_access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    .dst_access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT };
  timestamp_queries queries{ ctx->device() };
  VkQueryPoolCreateInfo query_info{};
  query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  query_info.queryCount = k_query_count;
  std::uint32_t family_count{};
  vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device(), &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device(), &family_count, families.data());
  auto const timestamp_bits = families.at(ctx->compute_queue_ref().family).timestampValidBits;
  if (timestamp_bits != 0) { (void)vkCreateQueryPool(ctx->device(), &query_info, nullptr, &queries.pool); }
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(ctx->physical_device(), &properties);
  state.SetLabel(static_cast<char const *>(properties.deviceName));
  // Count owned submit scopes, including the empty-graph scope and any DAG join.
  auto scopes_per_execution = std::max(std::size_t{ 1 }, prepared->execution.batches.size());
  if (!graph.step_predecessors.empty() && vkexec::detail::terminal_batches(prepared->execution).size() > 1) {
    ++scopes_per_execution;
  }
  auto const metrics = run_iterations<Mode>(state,
    recording_fixture{ .ctx = ctx.get(),
      .buffer = buffer->buffer,
      .bind = &compute->bind,
      .count = count,
      .prepared = &*prepared,
      .barrier = &barrier,
      .queries = &queries,
      .timestamp_bits = timestamp_bits,
      .timestamp_period = properties.limits.timestampPeriod,
      .scopes_per_execution = scopes_per_execution });
  state.counters.insert_or_assign(
    "temporary_allocations", static_cast<double>(metrics.allocations) / static_cast<double>(state.iterations()));
  if constexpr (Mode != mode::planning) {
    if (queries.pool != VK_NULL_HANDLE) {
      state.counters.insert_or_assign("gpu_ns", metrics.gpu_ns / static_cast<double>(state.iterations()));
    }
  }
  state.counters.insert_or_assign("passes", static_cast<double>(count));
  state.counters.insert_or_assign("barriers", static_cast<double>(barrier_count));
  state.counters.insert_or_assign(
    "recording_submissions", static_cast<double>(metrics.submitted_scopes) / static_cast<double>(state.iterations()));
  state.counters.insert_or_assign("descriptor_allocations", 0.0);
}
constexpr std::int64_t k_small_pass_count = 8;
constexpr std::int64_t k_medium_pass_count = 32;
constexpr std::int64_t k_large_pass_count = 128;
constexpr std::int64_t k_dense_pass_count = 512;
BENCHMARK_TEMPLATE(recording, mode::manual)
  ->Name("ManualRecording")->Arg(k_small_pass_count)->Arg(k_medium_pass_count)->Arg(k_large_pass_count)->Arg(k_dense_pass_count);
BENCHMARK_TEMPLATE(recording, mode::owned)
  ->Name("OwnedRecorder")->Arg(k_small_pass_count)->Arg(k_medium_pass_count)->Arg(k_large_pass_count)->Arg(k_dense_pass_count);
BENCHMARK_TEMPLATE(recording, mode::external_prepared)
  ->Name("ExternalRecorder")->Arg(k_small_pass_count)->Arg(k_medium_pass_count)->Arg(k_large_pass_count)->Arg(k_dense_pass_count);
BENCHMARK_TEMPLATE(recording, mode::external)
  ->Name("RecordInto")->Arg(k_small_pass_count)->Arg(k_medium_pass_count)->Arg(k_large_pass_count)->Arg(k_dense_pass_count);
BENCHMARK_TEMPLATE(recording, mode::planning)
  ->Name("GraphPlanning")->Arg(k_small_pass_count)->Arg(k_medium_pass_count)->Arg(k_large_pass_count)->Arg(k_dense_pass_count);
BENCHMARK_TEMPLATE(recording, mode::owned_execution)
  ->Name("OwnedExecution")->Arg(k_small_pass_count)->Arg(k_medium_pass_count)->Arg(k_large_pass_count)->Arg(k_dense_pass_count);
BENCHMARK_TEMPLATE(recording, mode::owned_dag_execution)
  ->Name("OwnedExecutionDag")
  ->Arg(k_small_pass_count)
  ->Arg(k_medium_pass_count)
  ->Arg(k_large_pass_count)
  ->Arg(k_dense_pass_count);
}// namespace
