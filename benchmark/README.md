# Pass graph benchmarks

Configure with `-Dvkexec_BUILD_BENCHMARKS=ON` and run `vkexec_benchmarks` from a Release build. The benchmark target is not installed or included in normal tests.

The benchmark families are split into separate translation units so they can compile in parallel.

All graph benchmarks use 5, 50, and 500 steps. The `lifecycle` cases include construction and destruction. The static case constructs the tuple representation; the dynamic case uses `make_dynamic_pass_graph` and `append`. They are therefore not equivalent public API construction measurements. The `record` cases reuse storage, and `build_record` includes construction, recording, and destruction. All recording uses steps that do not issue Vulkan commands; a context facade with no device is sufficient for these cases.

The dynamic `lifecycle` cases report allocations and allocated bytes per graph, counted on the benchmark thread during graph construction. The counters use benchmark executable replacements for ordinary `operator new` and `operator new[]`; allocations using aligned allocation or `malloc` directly are outside that count.

For baseline measurements, use a fixed CPU and stable performance governor, then run with `--benchmark_repetitions=10 --benchmark_report_aggregates_only=true`. CPU frequency scaling makes the smallest recording measurements noisy.

Initial measurements indicated that dynamic pass graph lifecycle cost was dominated by per-step allocation rather than virtual dispatch or vector growth. The public `lifecycle/dynamic`, `record/dynamic`, and `build_record/dynamic` cases now exercise graph-owned arena storage through `make_dynamic_pass_graph` and `append`. Rerun these cases to validate the production implementation. Static `record` results are especially prone to compiler specialization; compare absolute dynamic time per step rather than ratios.

The `heap_*` cases reproduce the former per-step `unique_ptr` and virtual-model representation in the benchmark executable. The `compact_*` cases preserve the benchmark-only 64-byte operation-table experiment, and the `arena_*` cases preserve the benchmark-only graph-owned PMR prototype. All three use the same noop, representative, and oversized steps at 5, 50, and 500 elements, with reserved and unreserved lifecycle, reused-graph recording, and reserved build-plus-record cases. Run `heap_reserved/representative/500` alongside `arena_reserved/representative/500` and `lifecycle/dynamic/representative/500` in the same build. The prototype arena reports `upstream_allocs/graph` and `upstream_bytes/graph` separately because aligned allocations made by the PMR upstream resource may bypass the benchmark's global `operator new` counter.
