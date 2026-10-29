# Resource-range mutation baseline

This is the historical, **experimental 4/47 audit**, not a comprehensive range
audit. The coverage-attribution investigation identified a raw-profile attribution limitation and measured a
[corrected indexed-profile baseline](mutation-coverage-attribution.md#corrected-baseline-and-limits).
The original measurements below are preserved.

Measured on 2026-10-29, x86_64-linux, using `nix develop .#clang-mut`.
Source commit: `075aa658318cb90ad206159f7144a3f1d0147e58`, plus the initial integration’s
validator, reporting and compiler-launcher changes. The range implementation,
selected tests, flake lock and `mull.yml` were unchanged. This is a working-tree
measurement, not an exact committed baseline (`source_dirty: true`,
`finalized_source_commit: null` in JSON). After committing the initial integration, rerun both audits
from that clean revision and record its full hash and new timings in a follow-up
baseline record. Do not substitute a later commit hash for this measurement. Machine-readable counts
and per-file/per-mutator statistics are in [mutation-baseline.json](mutation-baseline.json).

Toolchain: Clang **22.1.8**, Mull **0.35.0**, Mull LLVM **22.1.8**.
Preset: `unixlike-clang-mut`, Debug, `-O0`, timeout 10000 ms per mutant.
Both `vkexec` and `vkexec_core_tests` were recompiled with
`-fpass-plugin=$MULL_PLUGIN` and the build-directory `MULL_CONFIG`, bypassing
compiler caching. Compilation, linking, original tests, mutation execution and
SQLite validation all succeeded with the pinned Nix frontend and runner.

Selected executable/filter: `test/vkexec_core_tests '[ranges]'`.
The original suite passed **72,918 assertions in 53 test cases** before each
audit. These tests do not create a Vulkan device; no physical GPU is required.
The Vulkan loader is still a runtime dependency and is supplied through the
existing `--ld-search-path` configuration.

| Mode | Total reported | Killed | Survived | Uncovered | Timed out | Crashed | Abnormal exit | Score | Runner time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Covered (`ranges`) | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 100% (4/4) | 244 ms |
| Include uncovered (`ranges_uncovered`) | 47 | 4 | 0 | 43 | 0 | 0 | 0 | 100% (4/4) | 234 ms |

Times are Mull's reported total execution times, excluding CMake configuration,
compilation and the separate original-test invocation. An earlier pair of runs
also produced identical counts (258 ms and 262 ms respectively).

The score is `100 * killed / (killed + survived)`. Timeouts, crashes, abnormal
exits and uncovered mutants are excluded from the denominator and remain
visible in summaries. Keep these two populations separate despite their equal
scores. The 43 uncovered mutants are 91.5% of the include-uncovered population;
100% here describes only four evaluated mutations, not complete range coverage.
Mull reports uncovered locations even in helpers called by the range tests, so
classify those locations and inspect coverage attribution before assuming that
all 43 represent missing test inputs. No equivalent-mutant exclusions or survivor
classifications were made in the initial audit; there were no reported survivors.

Scope remains only `include/vkexec/detail/resource_ranges.hpp`, using
`cxx_comparison`, `cxx_boundary` and `cxx_arithmetic`. Tests and fetched dependencies
are excluded by `mull.yml`; other modules and GPU execution are outside this audit.
No score threshold or CI job is introduced.

To reproduce from the repository root:

```sh
nix develop .#clang-mut
clang++ --version
mull-runner-22 --version
cmake --preset unixlike-clang-mut
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_ranges --verbose
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_ranges_uncovered
ctest --test-dir out/build/unixlike-clang-mut -R vkexec.mull_report_validator --output-on-failure
```

The targets write separate `.sqlite`, `.txt` and `.json` reports under
`out/build/unixlike-clang-mut/mutation-reports/`, named `vkexec_core_ranges`
and `vkexec_core_ranges_uncovered`. Each invocation removes its prior SQLite,
JSON and text outputs before running. The real reports verified Mull's `mutant`
schema (`filename`, `directory`, `execution_status`, `mutator`) and observed status
1 (killed) and 8 (uncovered); fixture tests cover surviving, timeout, crash and
abnormal statuses, score arithmetic, grouping, JSON output and rejected reports.
