#include "../test/support/recording_buffer.hpp"
#include "../test/support/recording_compute.hpp"
#include <benchmark/benchmark.h>
#include <vkexec/barrier.hpp>
#include <vkexec/record_into.hpp>
#include <vkexec/sync_wait.hpp>

namespace recording_allocations {
auto begin() noexcept -> void;
auto end() noexcept -> std::size_t;
}// namespace recording_allocations
namespace {
struct timestamp_queries
{
  VkDevice device{};
  VkQueryPool pool{};
  ~timestamp_queries()
  {
    if (pool) { vkDestroyQueryPool(device, pool, nullptr); }
  }
};
struct compute_step
{
  decltype(vkexec::uses(vkexec::buffer_use{})) resources;
  vkexec::compute_bind bind;
  VkQueryPool queries{};
  bool first{}, last{};
  auto record(vkexec::context &, VkCommandBuffer cmd, vkexec::detail::pass_cleanup &) -> vkexec::status
  {
    if (queries && first) {
      vkCmdResetQueryPool(cmd, queries, 0, 2);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
    }
    // record_pass returns void, matching vkCmdDispatch; it has no recording status.
    vkexec::record_pass(cmd, bind, nullptr, 0, vkexec::dispatch{ 1, 1, 1 });
    if (queries && last) { vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1); }
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
      compute_step{ vkexec::uses(vkexec::read_write(
                      buffer, 0, vkexec::test::recording_buffer::byte_size, vkexec::buffer_usage::storage_compute)),
        bind,
        queries,
        index == 0,
        index + 1 == count });
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

enum class mode { manual, owned, external_prepared, external, planning, owned_execution, owned_dag_execution };

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
  auto compute_result = vkexec::test::recording_compute::make(*ctx, buffer->buffer, buffer->byte_size);
  if (!compute_result) {
    state.SkipWithError(compute_result.error().message());
    return;
  }
  auto compute = std::move(*compute_result);
  constexpr bool executes_owned = Mode == mode::owned_execution || Mode == mode::owned_dag_execution;
  constexpr bool dag = Mode == mode::owned_dag_execution;
  auto graph = make_graph(*ctx, buffer->buffer, compute->bind, count, VK_NULL_HANDLE, dag);
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
    .size = buffer->byte_size,
    .src_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .src_access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    .dst_access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT };
  timestamp_queries queries{ ctx->device(), VK_NULL_HANDLE };
  VkQueryPoolCreateInfo query_info{};
  query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  query_info.queryCount = 2;
  std::uint32_t family_count{};
  vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device(), &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device(), &family_count, families.data());
  auto const timestamp_bits = families.at(ctx->compute_queue_ref().family).timestampValidBits;
  if (timestamp_bits != 0) { (void)vkCreateQueryPool(ctx->device(), &query_info, nullptr, &queries.pool); }
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(ctx->physical_device(), &properties);
  state.SetLabel(properties.deviceName);
  double gpu_ns{};
  std::size_t allocations{};
  std::size_t submitted_scopes{};
  // Count owned submit scopes, including the empty-graph scope and any DAG join.
  auto scopes_per_execution = std::max(std::size_t{ 1 }, prepared->execution.batches.size());
  if (!graph.step_predecessors.empty() && vkexec::detail::terminal_batches(prepared->execution).size() > 1) {
    ++scopes_per_execution;
  }
  for (auto _ : state) {
    state.PauseTiming();
    auto input =
      make_graph(*ctx, buffer->buffer, compute->bind, count, executes_owned ? queries.pool : VK_NULL_HANDLE, dag);
    if constexpr (Mode == mode::planning) {
      state.ResumeTiming();
      recording_allocations::begin();
      auto plan = vkexec::detail::prepare_graph(
        *ctx, input.steps, input.step_queues, input.step_predecessors, ctx->compute_queue_ref());
      allocations += recording_allocations::end();
      benchmark::DoNotOptimize(plan);
    } else if constexpr (executes_owned) {
      state.ResumeTiming();
      recording_allocations::begin();
      auto done = vkexec::try_sync_wait(std::move(input));
      allocations += recording_allocations::end();
      if (done.failed()) {
        state.SkipWithError("owned execution failed");
        break;
      }
      state.PauseTiming();
      submitted_scopes += scopes_per_execution;
      if (queries.pool) {
        std::array<std::uint64_t, 2> ticks{};
        if (vkGetQueryPoolResults(ctx->device(),
              queries.pool,
              0,
              2,
              sizeof(ticks),
              ticks.data(),
              sizeof(ticks.front()),
              VK_QUERY_RESULT_64_BIT)
            == VK_SUCCESS) {
          auto const mask = timestamp_bits == 64 ? ~std::uint64_t{} : (std::uint64_t{ 1 } << timestamp_bits) - 1;
          gpu_ns += static_cast<double>((ticks.back() - ticks.front()) & mask)
                    * static_cast<double>(properties.limits.timestampPeriod);
        }
      }
      state.ResumeTiming();
    } else {
      auto opened = vkexec::detail::submit_scope::open(*ctx);
      if (!opened) {
        state.SkipWithError(opened.error().message());
        break;
      }
      auto scope = vkexec::expected_take(opened);
      vkexec::result<vkexec::recorded_graph> recorded = vkexec::fail(vkexec::errc::empty_result);
      vkexec::status status;
      if (queries.pool) {
        vkCmdResetQueryPool(scope.cmd, queries.pool, 0, 2);
        vkCmdWriteTimestamp(scope.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries.pool, 0);
      }
      state.ResumeTiming();
      recording_allocations::begin();
      if constexpr (Mode == mode::manual) {
        for (std::size_t index{}; index < count; ++index) {
          if (index != 0) { status = vkexec::buffer_barrier(*ctx, scope.cmd, barrier); }
          if (!status) { break; }
          vkexec::record_pass(scope.cmd, compute->bind, nullptr, 0, vkexec::dispatch{ 1, 1, 1 });
        }
      } else if constexpr (Mode == mode::owned) {
        auto const info = vkexec::detail::recording_info(scope.queue);
        for (auto const &batch : prepared->execution.batches) {
          status = vkexec::detail::record_sync_batch(
            *ctx, scope.cmd, scope.cleanup, info, batch, prepared->synchronization, input.steps);
          if (!status) { break; }
        }
      } else if constexpr (Mode == mode::external_prepared) {
        status = vkexec::detail::record_external(*ctx, scope.cmd, scope.queue, *prepared, input.steps, scope.cleanup);
      } else {
        recorded = vkexec::record_into(*ctx, scope.cmd, std::move(input), { scope.queue });
      }
      allocations += recording_allocations::end();
      state.PauseTiming();
      if (queries.pool) { vkCmdWriteTimestamp(scope.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries.pool, 1); }
      if (!status || (Mode == mode::external && !recorded)) {
        (void)vkResetCommandBuffer(scope.cmd, 0);
        state.SkipWithError("recording failed");
        state.ResumeTiming();
        break;
      }
      auto const ended = scope.end_recording();
      if (!ended) {
        // vkEndCommandBuffer failed before any queue submission.
        if (vkResetCommandBuffer(scope.cmd, 0) != VK_SUCCESS) { std::terminate(); }
        if (recorded) {
          auto abandoned = recorded->abandon();
          if (!abandoned) { std::terminate(); }
        }
        state.SkipWithError(ended.error().message());
        state.ResumeTiming();
        break;
      }
      auto const submitted = ctx->submit_and_wait(scope.cmd, scope.queue);
      if (queries.pool && submitted) {
        std::array<std::uint64_t, 2> ticks{};
        if (vkGetQueryPoolResults(ctx->device(),
              queries.pool,
              0,
              2,
              sizeof(ticks),
              ticks.data(),
              sizeof(ticks.front()),
              VK_QUERY_RESULT_64_BIT)
            == VK_SUCCESS) {
          auto const mask = timestamp_bits == 64 ? ~std::uint64_t{} : (std::uint64_t{ 1 } << timestamp_bits) - 1;
          gpu_ns += static_cast<double>((ticks.back() - ticks.front()) & mask)
                    * static_cast<double>(properties.limits.timestampPeriod);
        }
      }
      if (recorded) {
        // submit_and_wait can fail after the queue has accepted the buffer.
        // Do not abandon work that may still be in flight.
        if (!submitted) { std::terminate(); }
        auto marked = recorded->mark_submitted();
        auto completed = marked ? recorded->complete() : marked;
        if (!completed) { state.SkipWithError(completed.error().message()); }
      }
      if (!submitted) { state.SkipWithError("submission failed"); }
      scope.release();
      state.ResumeTiming();
    }
    state.PauseTiming();
    input = vkexec::dynamic_pass_graph_sender{};
    state.ResumeTiming();
  }
  state.counters["temporary_allocations"] = static_cast<double>(allocations) / static_cast<double>(state.iterations());
  if constexpr (Mode != mode::planning) {
    if (queries.pool) { state.counters["gpu_ns"] = gpu_ns / static_cast<double>(state.iterations()); }
  }
  state.counters["passes"] = static_cast<double>(count);
  state.counters["barriers"] = static_cast<double>(barrier_count);
  state.counters["recording_submissions"] =
    static_cast<double>(submitted_scopes) / static_cast<double>(state.iterations());
  state.counters["descriptor_allocations"] = 0;
}
#define RECORDING_BENCHMARK(label, value) \
  BENCHMARK_TEMPLATE(recording, mode::value)->Name(label)->Arg(8)->Arg(32)->Arg(128)->Arg(512)
RECORDING_BENCHMARK("ManualRecording", manual);
RECORDING_BENCHMARK("OwnedRecorder", owned);
RECORDING_BENCHMARK("ExternalRecorder", external_prepared);
RECORDING_BENCHMARK("RecordInto", external);
RECORDING_BENCHMARK("GraphPlanning", planning);
RECORDING_BENCHMARK("OwnedExecution", owned_execution);
RECORDING_BENCHMARK("OwnedExecutionDag", owned_dag_execution);
#undef RECORDING_BENCHMARK
}// namespace
