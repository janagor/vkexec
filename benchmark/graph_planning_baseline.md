# Graph-planning baseline

Capture the baseline from a source tree containing only PR 1 changes. Record
the commit SHA of that exact tree. Run later revisions on the same machine with
equivalent build options, preferably on the same CPU core under low load.

```sh
cmake -S . -B build-bench-pr1 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -Dvkexec_BUILD_BENCHMARKS=ON \
  -Dvkexec_ENABLE_SANITIZER_ADDRESS=OFF \
  -Dvkexec_ENABLE_SANITIZER_UNDEFINED=OFF \
  -Dvkexec_ENABLE_CLANG_TIDY=OFF \
  -Dvkexec_ENABLE_CPPCHECK=OFF
cmake --build build-bench-pr1 --target vkexec_graph_planning_benchmark
mkdir -p benchmark/baselines
./build-bench-pr1/benchmark/vkexec_graph_planning_benchmark \
  --benchmark_repetitions=10 \
  --benchmark_out=benchmark/baselines/pr1-baseline.json \
  --benchmark_out_format=json
```

Keep individual repetitions in the JSON; do not pass
`--benchmark_report_aggregates_only=true`.
The JSON context records the CPU, date, load, and executable path. Record these
additional values alongside the new JSON:

| Field | Value |
| --- | --- |
| Exact source commit SHA | Pending clean Release run |
| Compiler and version | Pending clean Release run |
| Release optimization flags | Pending clean Release run |
| CPU model | Pending clean Release run |
| Address and undefined sanitizers | Disabled for the requested baseline |
| Benchmark command and repetitions | Command above, 10 repetitions |

The earlier root-level `baseline.json` was built as `RelWithDebInfo` with
address and undefined sanitizers enabled. It is excluded from PR 1 and is not
the definitive baseline.
