# Pass graph benchmarks

Configure with `-Dvkexec_BUILD_BENCHMARKS=ON` and run `vkexec_benchmarks` from a Release build. The benchmark target is not installed or included in normal tests.

The `lifecycle` cases include construction and destruction. The static case constructs the tuple representation; the dynamic case uses `make_dynamic_pass_graph` and `append`. They are therefore not equivalent public API construction measurements. The `record` cases reuse storage, and `build_record` includes construction, recording, and destruction. All recording uses steps that do not issue Vulkan commands; a context facade with no device is sufficient for these cases.

The dynamic `lifecycle` cases report allocations and allocated bytes per graph, counted on the benchmark thread during graph construction. The counters use benchmark executable replacements for ordinary `operator new` and `operator new[]`; allocations using aligned allocation or `malloc` directly are outside that count.

For baseline measurements, use a fixed CPU and stable performance governor, then run with `--benchmark_repetitions=10 --benchmark_report_aggregates_only=true`. CPU frequency scaling makes the smallest recording measurements noisy.

Initial measurements indicate that dynamic pass graph lifecycle cost is dominated by per-step allocation rather than virtual dispatch or vector growth. Static `record` results are especially prone to compiler specialization; compare absolute dynamic time per step rather than ratios.

The `sbo_reserved`, `sbo_no_reserve`, and `sbo_record` cases compare the original heap representation (`0`) with 16, 32, and 64 byte inline storage. They construct a vector directly so that each capacity can be tested in one executable. The `oversized` step exercises heap fallback. Lifecycle cases report `step_bytes`, `model_bytes`, and `model_align` as well as allocation counts. These are experimental measurements; the public dynamic graph continues to use the original heap representation until a capacity is selected from results.

The `compact_*` cases use a benchmark-only operation table and a 64-byte inline buffer. The `arena_*` cases store one pointer per vector entry and allocate polymorphic models from a graph-owned monotonic resource. Both use the same noop, representative, and oversized steps at 1000 elements, with reserved and unreserved lifecycle, reused-graph recording, and reserved build-plus-record cases. The arena reports `upstream_allocs/graph` and `upstream_bytes/graph` separately because aligned allocations made by the PMR upstream resource may bypass the benchmark's global `operator new` counter. These candidates do not change the public graph representation.
