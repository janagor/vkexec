#include "../test/support/graph_submission_reference.hpp"

#include <vkexec/context.hpp>
#include <vkexec/detail/graph_submission_sync.hpp>
#include <vkexec/detail/pass_execution.hpp>
#include <vkexec/detail/pass_graph_submit.hpp>
#include <vkexec/result.hpp>
#include <vkexec/submit_scope.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_features/timeline_semaphore.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace {

constexpr std::int64_t k_small_nodes = 8;
constexpr std::int64_t k_medium_nodes = 32;
constexpr std::int64_t k_large_nodes = 128;
constexpr std::int64_t k_dense_nodes = 512;
constexpr std::size_t k_queue_count = 2;
constexpr int k_node_argument = 0;
constexpr int k_shape_argument = 1;
constexpr int k_mode_argument = 2;
constexpr std::int64_t k_binary_mode = 0;
constexpr std::int64_t k_timeline_mode = 1;

struct submission_counters
{
  std::size_t batches{};
  std::size_t dependency_edges{};
  std::size_t waits{};
  std::size_t signals{};
  std::size_t semaphores{};
};

[[nodiscard]] auto collect_counters(vkexec::detail::execution_plan const &plan,
  vkexec::detail::prepared_batch_sync const &prepared,
  bool timeline) -> submission_counters
{
  auto const edges = vkexec::detail::graph_dependency_count(plan);
  return submission_counters{ .batches = plan.batches.size(),
    .dependency_edges = edges,
    .waits = prepared.waits.size(),
    .signals = prepared.signals.size(),
    .semaphores = timeline ? plan.batches.size() : edges };
}

auto set_counters(benchmark::State &state, submission_counters const &counters) -> void
{
  state.counters.insert_or_assign("batches", static_cast<double>(counters.batches));
  state.counters.insert_or_assign("dependency_edges", static_cast<double>(counters.dependency_edges));
  state.counters.insert_or_assign("waits", static_cast<double>(counters.waits));
  state.counters.insert_or_assign("signals", static_cast<double>(counters.signals));
  state.counters.insert_or_assign("semaphores", static_cast<double>(counters.semaphores));
}

[[nodiscard]] auto has_distinct_graphics_queue(vkexec::context const &ctx) -> bool
{
  auto const graphics = ctx.graphics_queue_ref();
  return graphics.queue != VK_NULL_HANDLE && graphics.queue != ctx.compute_queue_ref().queue;
}

[[nodiscard]] auto make_execution_fixture(vkexec::context &ctx, std::size_t count, vkexec::test::submission_shape shape)
  -> vkexec::result<vkexec::detail::graph_submit_sender>
{
  auto plan = vkexec::test::submission_plan(count, shape);
  for (auto &batch : plan.batches) {
    bool const use_graphics =
      shape == vkexec::test::submission_shape::mixed_queue && batch.first_step % k_queue_count != 0;
    batch.queue = use_graphics ? ctx.graphics_queue_ref() : ctx.compute_queue_ref();
  }
  // Match production retirement: all independent sinks feed a terminal join.
  auto sinks = vkexec::detail::terminal_batches(plan);
  if (sinks.size() > std::size_t{ 1 }) {
    vkexec::detail::execution_batch join;
    join.queue = ctx.compute_queue_ref();
    join.predecessors = std::move(sinks);
    plan.batches.push_back(std::move(join));
  }
  std::vector<vkexec::detail::submit_scope> scopes;
  scopes.reserve(plan.batches.size());
  for (auto &batch : plan.batches) {
    auto opened = vkexec::detail::submit_scope::open(ctx, batch.queue);
    if (!opened) { return vkexec::fail(opened); }
    scopes.push_back(vkexec::expected_take(opened));
    VKEXEC_TRY(scopes.back().end_recording());
  }
  return vkexec::detail::graph_submit_sender{
    .scopes = std::move(scopes), .plan = std::move(plan), .presentation = {}
  };
}

[[nodiscard]] auto execute_graph(vkexec::context &ctx, std::size_t count, vkexec::test::submission_shape shape)
  -> vkexec::status
{
  auto fixture = make_execution_fixture(ctx, count, shape);
  if (!fixture) { return vkexec::fail(fixture); }
  auto outcome = vkexec::try_sync_wait(std::move(*fixture));
  if (outcome.failed()) { return vkexec::fail(outcome.take_error()); }
  return {};
}

template<bool Reference> auto prepare_submission(benchmark::State &state) -> void
{
  auto const plan = vkexec::test::submission_plan(static_cast<std::size_t>(state.range(k_node_argument)),
    static_cast<vkexec::test::submission_shape>(state.range(k_shape_argument)));
  bool const timeline = state.range(k_mode_argument) == k_timeline_mode;
  std::vector<VkSemaphore> handles(timeline ? plan.batches.size() : vkexec::detail::graph_dependency_count(plan));
  auto prepare = [&]() -> vkexec::detail::prepared_batch_sync {
    if constexpr (Reference) {
      return vkexec::test::reference_batch_sync(plan, handles, timeline);
    } else {
      return vkexec::detail::prepare_batch_sync(plan, handles, timeline);
    }
  };
  auto const prepared = prepare();
  // NOLINTNEXTLINE(readability-identifier-length) -- Google Benchmark keep-running idiom
  for ([[maybe_unused]] auto _ : state) {
    auto result = prepare();
    benchmark::DoNotOptimize(result);
    benchmark::ClobberMemory();
  }
  set_counters(state, collect_counters(plan, prepared, timeline));
}

auto bm_prepare_submission(benchmark::State &state) -> void { prepare_submission<false>(state); }
auto bm_prepare_submission_reference(benchmark::State &state) -> void { prepare_submission<true>(state); }

[[nodiscard]] auto benchmark_options(bool timeline) -> vkexec::scheduler_options
{
  vkexec::scheduler_options options;
  if (timeline) { vkexec::feat::feature_traits<vkexec::feat::timeline_semaphore>::configure(options.requirements); }
  return options;
}

[[nodiscard]] auto expensive_gpu_enabled() -> bool
{
  // NOLINTNEXTLINE(concurrency-mt-unsafe) -- benchmark process does not modify the environment
  return std::getenv("VKEXEC_BENCH_EXPENSIVE_GPU") != nullptr;
}

// Minimal real command buffers exercise the production graph submission sender.
// Context creation and one warm-up execution are outside the timed loop.
auto bm_graph_execution(benchmark::State &state) -> void
{
  auto const count = static_cast<std::size_t>(state.range(k_node_argument));
  auto const shape = static_cast<vkexec::test::submission_shape>(state.range(k_shape_argument));
  if (count == static_cast<std::size_t>(k_dense_nodes) && !expensive_gpu_enabled()) {
    state.SkipWithMessage("set VKEXEC_BENCH_EXPENSIVE_GPU=1 to enable 512-batch GPU cases");
    return;
  }
  bool const timeline = state.range(k_mode_argument) == k_timeline_mode;
  auto created = vkexec::try_sync_wait_value(vkexec::factory::make_context(benchmark_options(timeline)));
  if (!created) {
    state.SkipWithError(created.error().message());
    return;
  }
  auto &ctx = **created;
  if (ctx.capabilities().timeline_semaphore != timeline) {
    state.SkipWithError("context synchronization mode does not match requested mode");
    return;
  }
  state.counters.insert_or_assign("timeline", timeline ? 1.0 : 0.0);
  if (shape == vkexec::test::submission_shape::mixed_queue && !has_distinct_graphics_queue(ctx)) {
    state.SkipWithMessage(
      "default compute-only context has no distinct graphics queue; multi-queue GPU coverage unavailable");
    return;
  }
  if (auto warmed = execute_graph(ctx, count, shape); !warmed) {
    state.SkipWithError(warmed.error().message());
    return;
  }
  // NOLINTNEXTLINE(readability-identifier-length) -- Google Benchmark keep-running idiom
  for ([[maybe_unused]] auto _ : state) {
    if (auto executed = execute_graph(ctx, count, shape); !executed) {
      state.SkipWithError(executed.error().message());
      break;
    }
  }
}

// Stage measurements exclude recording, completion waiting, and destruction.
// The separate execution benchmark measures the asynchronous terminal path.
struct stage_execution
{
  explicit stage_execution(vkexec::context &context) : ctx(&context) {}
  stage_execution(stage_execution const &) = delete;
  auto operator=(stage_execution const &) -> stage_execution & = delete;
  stage_execution(stage_execution &&) = delete;
  auto operator=(stage_execution &&) -> stage_execution & = delete;

  vkexec::context *ctx;
  vkexec::detail::graph_submit_sender::op_state<int> operation{};
  std::size_t submitted{};

  ~stage_execution()
  {
    if (submitted != 0) {
      constexpr std::size_t k_retirement_attempts = 3;
      bool retired{};
      for (std::size_t attempt{}; attempt < k_retirement_attempts; ++attempt) {
        auto const result = vkWaitForFences(
          ctx->device(), static_cast<std::uint32_t>(submitted), operation.fences.data(), VK_TRUE, UINT64_MAX);
        if (result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST) {
          retired = true;
          break;
        }
      }
      if (!retired) { std::terminate(); }
    }
    operation.release(ctx->device());
  }
};

auto bm_submission_stages(benchmark::State &state) -> void
{
  auto const count = static_cast<std::size_t>(state.range(k_node_argument));
  auto const shape = static_cast<vkexec::test::submission_shape>(state.range(k_shape_argument));
  if (count == static_cast<std::size_t>(k_dense_nodes) && !expensive_gpu_enabled()) {
    state.SkipWithMessage("set VKEXEC_BENCH_EXPENSIVE_GPU=1 to enable 512-batch GPU cases");
    return;
  }
  bool const timeline = state.range(k_mode_argument) == k_timeline_mode;
  auto created = vkexec::try_sync_wait_value(vkexec::factory::make_context(benchmark_options(timeline)));
  if (!created) {
    state.SkipWithError(created.error().message());
    return;
  }
  auto &ctx = **created;
  if (ctx.capabilities().timeline_semaphore != timeline) {
    state.SkipWithError("context synchronization mode does not match requested mode");
    return;
  }
  state.counters.insert_or_assign("timeline", timeline ? 1.0 : 0.0);
  if (shape == vkexec::test::submission_shape::mixed_queue && !has_distinct_graphics_queue(ctx)) {
    state.SkipWithMessage(
      "default compute-only context has no distinct graphics queue; multi-queue GPU coverage unavailable");
    return;
  }
  // Complete one graph before starting either benchmark or stage timing.
  if (auto warmed = execute_graph(ctx, count, shape); !warmed) {
    state.SkipWithError(warmed.error().message());
    return;
  }
  using clock = std::chrono::steady_clock;
  double object_seconds{};
  double prepare_seconds{};
  double submit_seconds{};
  submission_counters counters;
  std::size_t fences_created{};
  std::size_t semaphores_created{};
  std::size_t submit_calls{};
  // NOLINTNEXTLINE(readability-identifier-length) -- Google Benchmark keep-running idiom
  for ([[maybe_unused]] auto _ : state) {
    state.PauseTiming();
    {
      auto fixture = make_execution_fixture(ctx, count, shape);
      if (!fixture) {
        state.SkipWithError(fixture.error().message());
        return;
      }
      stage_execution execution{ ctx };
      auto &operation = execution.operation;
      operation.plan = std::move(fixture->plan);
      operation.scopes = std::move(fixture->scopes);
      state.ResumeTiming();
      auto const object_start = clock::now();
      auto const sync = operation.create_sync(ctx);
      auto const prepare_start = clock::now();
      if (!sync) {
        state.SkipWithError(sync.error().message());
        return;
      }
      operation.batch_sync =
        vkexec::detail::prepare_batch_sync(operation.plan, operation.semaphores, operation.timeline);
      auto const submit_start = clock::now();
      for (std::size_t batch{}; batch < operation.scopes.size(); ++batch) {
        if (auto submitted = operation.submit_batch(ctx, batch); !submitted) {
          state.SkipWithError(submitted.error().message());
          return;
        }
        ++execution.submitted;
      }
      auto const submit_end = clock::now();
      state.PauseTiming();
      object_seconds += std::chrono::duration<double>(prepare_start - object_start).count();
      prepare_seconds += std::chrono::duration<double>(submit_start - prepare_start).count();
      submit_seconds += std::chrono::duration<double>(submit_end - submit_start).count();
      counters = collect_counters(operation.plan, operation.batch_sync, operation.timeline);
      fences_created = operation.fences.size();
      semaphores_created = operation.semaphores.size();
      submit_calls = execution.submitted;
      // Destruction and GPU waits occur while timing is paused.
      // Explicit retirement here leaves the destructor as the failure guard.
      if (auto const waited = vkWaitForFences(ctx.device(),
            static_cast<std::uint32_t>(execution.submitted),
            operation.fences.data(),
            VK_TRUE,
            UINT64_MAX);
        waited != VK_SUCCESS) {
        state.SkipWithError("stage benchmark fence wait failed");
        return;
      }
      execution.submitted = 0;
      operation.release(ctx.device());
    }
    state.ResumeTiming();
  }
  set_counters(state, counters);
  state.counters.insert_or_assign("fences_created", static_cast<double>(fences_created));
  state.counters.insert_or_assign("semaphores_created", static_cast<double>(semaphores_created));
  state.counters.insert_or_assign("submit_calls", static_cast<double>(submit_calls));
  auto const iterations = static_cast<double>(state.iterations());
  state.counters.insert_or_assign("object_creation_seconds", object_seconds / iterations);
  state.counters.insert_or_assign("preparation_seconds", prepare_seconds / iterations);
  state.counters.insert_or_assign("queue_submission_seconds", submit_seconds / iterations);
}

auto register_cases(benchmark::Benchmark *entry) -> void
{
  constexpr std::array k_sizes{ k_small_nodes, k_medium_nodes, k_large_nodes, k_dense_nodes };
  constexpr std::array k_shapes{ vkexec::test::submission_shape::linear,
    vkexec::test::submission_shape::independent,
    vkexec::test::submission_shape::fork_join,
    vkexec::test::submission_shape::dense,
    vkexec::test::submission_shape::fan_in,
    vkexec::test::submission_shape::mixed_queue };
  for (auto const count : k_sizes) {
    for (auto const shape : k_shapes) {
      for (auto const mode : { k_binary_mode, k_timeline_mode }) {
        entry->Args({ count, static_cast<std::int64_t>(shape), mode });
      }
    }
  }
  entry->ArgNames({ "batches", "shape", "timeline" });
  entry->Unit(benchmark::kMicrosecond);
}

BENCHMARK(bm_prepare_submission_reference)->Apply(register_cases);
BENCHMARK(bm_prepare_submission)->Apply(register_cases);
BENCHMARK(bm_submission_stages)->Apply(register_cases)->UseRealTime();
BENCHMARK(bm_graph_execution)->Apply(register_cases)->UseRealTime();

}// namespace
