#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. Additional arguments are Google Benchmark flags.
# Override VKEXEC_BENCH_BINARY to use a different build directory.
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${VKEXEC_BENCH_BINARY:-$repo_dir/out/build/unixlike-clang-release/benchmark/vkexec_graph_submission_benchmark}"
result_dir="$repo_dir/out/bench-results"
mkdir -p -- "$result_dir"
result_file="$result_dir/submission-$(date -u +%Y%m%dT%H%M%SZ)-$$.json"

"$binary" \
  --benchmark_repetitions=7 \
  --benchmark_report_aggregates_only=true \
  "$@" \
  --benchmark_out="$result_file" \
  --benchmark_out_format=json
printf 'Results: %s\n' "$result_file"
