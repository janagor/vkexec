# Mutation testing

Mull audits whether the existing Catch2 tests detect deliberate implementation changes. Keep unit tests, coverage, sanitizer runs, and Vulkan integration tests alongside this audit.

The opt-in `unixlike-clang-mut` preset exposes all registered mutation audits. Each target orchestrates a persistent private build that instruments the core library and the selected test executable, including header-defined implementation code. The range YAML `mull.yml` selects only `include/vkexec/detail/resource_ranges.hpp`; the DAG YAML selects `include/vkexec/detail/pass_execution.hpp`. Tests and fetched dependencies are excluded by these source scopes. The targets select the device-independent `[ranges]` and `[dag-plan]` cases respectively.

## Toolchain and execution

On x86_64-linux and aarch64-linux, the `clang-mut` shell provides Mull 0.35.0 for LLVM 22.1.8. Only `clang-mut` includes mutation tools and exports their paths; the default, `clang`, and `clang16` shells do not include Mull:

```sh
nix develop .#clang-mut
cmake --preset unixlike-clang-mut
```

The `unixlike-clang-mut` preset is the only public mutation preset. It inherits the normal Clang debug toolchain, reads `MULL_PLUGIN` and `MULL_RUNNER` from the environment, and enables mutation testing. It disables hardening, IPO, coverage tooling, sanitizers, static analyzers, optional modules, and compiler caching for this narrow audit. Ordinary debug and release presets retain their original settings and do not enable Mull. If an existing build directory has mutation testing cached as enabled, configure it once with `-Dvkexec_ENABLE_MUTATION_TESTING=OFF` to clear that setting.

The `clang-mut` shell also disables Nix's fortify hardening, which requires optimization and conflicts with Mull's `-O0` instrumentation. After changing this shell setting, exit the old Nix shell and enter `clang-mut` again before configuring or building.

The shell exports `MULL_LIBRARY_PATH` for Vulkan and the C/C++ runtimes. CMake uses it to initialize the semicolon-separated `vkexec_MULL_LIBRARY_PATHS` cache variable and passes each directory to Mull with `--ld-search-path`. Mull 0.35's dependency scanner does not follow ELF `RUNPATH`, so it needs these paths even when the test executable runs successfully. After a toolchain change, clear this cache entry with `cmake --preset unixlike-clang-mut -Uvkexec_MULL_LIBRARY_PATHS` to refresh it from the shell.

`MULL_PLUGIN` contains the actual Nix store path to the frontend; `MULL_RUNNER` contains the runner path. The same arguments can be appended to a custom CMake configuration with mutation testing enabled. The shell repackages official Mull Debian binaries, pins their checksums, and patches runtime library paths to the Nix LLVM dependencies; it does not build Mull from source. The LLVM patch version is checked during evaluation so a toolchain update requires reviewing the package pin. Matching versions alone do not establish binary compatibility. Instrumented compilation and both audit targets have now been verified with this toolchain; see the [measured baseline](mutation-baseline.md).

Outside this shell, install a Mull frontend and runner matching the Clang toolchain being used. Supply the frontend path explicitly. CMake finds `mull-runner-<Clang major>` on `PATH`, or accepts an explicit `vkexec_MULL_RUNNER` path.

From the repository root, configure and run in a separate build directory:

```sh
clang++ --version
mull-runner-22 --version
cmake --preset unixlike-clang-mut -Dvkexec_MULL_PLUGIN=/absolute/path/to/mull-ir-frontend-22
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_ranges
```

Record these versions and confirm that `clang++` is the compiler selected in the CMake cache. A matching major-version suffix and an existing plugin file do not prove LLVM build compatibility. The instrumented compilation must successfully load the supplied plugin; inspect the verbose build (`cmake --build out/build/unixlike-clang-mut/mutation-builds/ranges --target vkexec_core_tests --verbose`) and Mull's configuration messages before accepting the toolchain. Plugin loading has been verified by recompiling both selected targets with this frontend and running their generated mutants.

The [baseline record](mutation-baseline.md) contains measured results and their limitations. Building this target compiles the instrumented tests and executes Mull. It is separate from ordinary CTest and is never part of the default build.

The preset disables IPO, hardening, sanitizers, coverage tooling, compiler caching, static analyzers, examples, and optional modules. Mull uses its own LLVM profiling flags for covered-code selection. Instrumentation is private to the two selected targets; third-party targets are not instrumented. CMake copies the scope YAML to the private build's `mull.yml` for compilation and `mutation-configs/<scope>.yml` for the runner, and explicitly sets `MULL_CONFIG` for both. The two copies have identical contents. A configuration digest changes the selected targets' compile definitions when the YAML changes, forcing mutant regeneration after CMake reconfiguration. Verify the two copied configurations match on the first real run.

The warmup/baseline timeout defaults to 10000 milliseconds, adjustable with `-Dvkexec_MULL_TIMEOUT_MS=...`. Mull 0.35 derives mutant timeouts from ten times the measured baseline duration, with an upstream default minimum of 30 ms (this project selects 250 ms); `--timeout` does not set a fixed per-mutant budget. Keep timeout outcomes separate when comparing scores.

## Review the baseline

Mull prints findings and writes `out/build/unixlike-clang-mut/mutation-reports/vkexec_core_ranges.sqlite`. Each target removes its previous reports and profiles, runs the unmodified `[ranges]` tests with a fresh per-audit `LLVM_PROFILE_FILE`, merges that raw profile with matching `llvm-profdata`, runs Mull with `--coverage-info` pointing to the indexed profile, and validates the report using Python's standard-library SQLite support. The `.profraw` and `.profdata` files remain beside the reports for inspection. This works around the [verified multi-TU raw-profile attribution limitation](mutation-coverage-attribution.md); both compiled targets retain the mutation pass and coverage instrumentation. Surviving mutants are allowed during baseline collection; build, original-test, runner, or report-validation errors fail the target.

The validator requires every explicitly expected source file to contain mutants and a report containing only those source paths and at least one killed or surviving mutant. It reports killed, survived, timed-out, crashed, abnormal-exit, and uncovered counts separately. The mutation score is `100 * killed / (killed + survived)`; timeouts, crashes, abnormal exits and uncovered mutants are excluded from this denominator, never counted as kills. Reports containing only these excluded outcomes fail validation. Missing/empty reports, unknown statuses, dry-run results, and unsupported schemas fail validation. Its schema and status definitions match locally generated Mull 0.35.0 reports and were inspected in [Mull's upstream source](https://github.com/mull-project/mull/blob/dab0b590af1ae1ec34609d993668a9a12a703c30/rust/mull-reporters/src/sqlite_reporter.rs), with killed and uncovered statuses observed in real runs. Other execution statuses remain covered by validator fixtures.

Inspect killed and surviving mutants with the matching reporter:

```sh
MULL_CONFIG="$PWD/out/build/unixlike-clang-mut/mutation-configs/ranges.yml" mull-reporter-22 \
  out/build/unixlike-clang-mut/mutation-reports/vkexec_core_ranges.sqlite --ide-reporter-show-killed
```

For a separate audit that also exposes uncovered mutants:

```sh
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_ranges_uncovered
```

This passes `--include-not-covered` and writes `vkexec_core_ranges_uncovered.sqlite`. Keep its results labeled separately from the covered-only audit; do not combine their scores. Both reports must contain killed or surviving mutants, even if uncovered mutants are present.

Each target removes its previous SQLite, JSON and text reports before execution. Each target also writes a sibling `.json` summary with schema version 2, overall counts, the score definition and denominator, per-file/per-mutator statistics, sorted per-location records (`mutant_id`, `file`, `line`, `column`, `mutator`, `status`), and evaluated/discovered diagnostic percentages. Evaluated includes killed, survived, timed-out, crashed and abnormal-exit outcomes; uncovered is excluded. This percentage is separate from the mutation score. The `file` fields are normalized relative to the repository root, independent of checkout location. The opaque `mutant_id` is preserved verbatim for correlation with SQLite; it may embed absolute paths and is therefore checkout-specific. It is not guaranteed to distinguish binary instances or translation units. `--repository-root` overrides the default root inferred from the validator location. Sources outside that root are rejected. Groups without killed or surviving mutants have a `null` score. To regenerate a summary independently:

```sh
python3 scripts/validate_mull_report.py \
  out/build/unixlike-clang-mut/mutation-reports/vkexec_core_ranges.sqlite \
  include/vkexec/detail/resource_ranges.hpp --json-output /tmp/ranges-summary.json
```

Instrumented targets bypass any cached global compiler launcher, including a launcher retained from an earlier configuration with compiler caching enabled.

Record the source revision, compiler and Mull versions, filters, test selection, killed/surviving/uncovered counts, and equivalent-mutant decisions with each baseline. Review survivors for missing boundary assertions or incorrect dependency expectations, then strengthen the existing tests and rerun with the same scope. Remove tests only after confirming that remaining tests preserve their behavioral coverage and mutation sensitivity. See the [baseline record](mutation-baseline.md) for the initial measured scores.

For future scopes, use the [scope registration API](mutation-audit-scopes.md).
All registered targets are available from the same public build directory without
changing a scope cache variable. Internal builds are necessary for Mull 0.35.0:
its coverage filter can retain out-of-scope uncovered mutants before runtime path
filtering, and runtime operator settings do not remove already-compiled arithmetic
mutants. The [small real experiment](mutation-dag-baseline.md#single-preset-fallback)
records why sharing one instrumented binary was rejected. Keep independent reports
separate; this integration does not enable a CI gate.

See Mull's [setup guide](https://mull-project.com/getting-started/setup/), [runner reference](https://mull-project.com/reference/mull-runner/), and [filters](https://mull-project.com/reference/filters/) for tool behavior and reporting options.

## Investigating uncovered locations

See the [source-coverage investigation](mutation-coverage-investigation.md) for the measured source-region discrepancy and reproduction commands. Do not infer missing assertions from Mull uncovered status alone.

The [coverage-attribution investigation](mutation-coverage-attribution.md) records automatic/raw/indexed comparisons, isolated instrumentation controls, the standalone reproduction, and the corrected baseline. The initial 4/47 audit remains experimental historical evidence. Indexed profiles evaluate all 47 discovered mutants; surviving and timed-out mutants remain visible, with no CI score gate.

## Reproducible range-quality audit

The [PR 4 quality measurements](mutation-range-quality.md) select one worker and
`vkexec_MULL_MINIMUM_TIMEOUT_MS=250`. Both cache values must be positive integers
and exist only in mutation-enabled configurations. Override them explicitly when
investigating another machine:

```sh
cmake --preset unixlike-clang-mut \
  -Dvkexec_MULL_WORKERS=1 -Dvkexec_MULL_MINIMUM_TIMEOUT_MS=250
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_ranges
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_ranges_uncovered
ctest --test-dir out/build/unixlike-clang-mut -R 'vkexec.mull_' --output-on-failure
```

The adaptive budget remains `max(10 * baseline, minimum timeout)`; the separate
warmup timeout remains 10000 ms. Catch2 receives `--order lex --abort` during both
original profile collection and mutation evaluation. Catch2 3.12 defaults to
randomized order: without an explicit order, a mutated tracker can enter expensive
region processing before reaching an assertion that detects the same defect.
Fail-fast exits on that real assertion failure. A mutation that hangs before any
assertion still times out and is reported separately. No timeout is converted to
a kill. The indexed-profile generation and coverage filtering are unchanged.

Before rebuilding, preserve the executable, its SHA256, copied `mull.yml`, source
revision and dirty diff, tool versions, CMake cache, and existing reports. Each
target overwrites its own reports: copy the entire `mutation-reports` directory
after **each** invocation into a separately named directory, including SQLite,
JSON, text, raw profile and indexed profile. Repeat each audit three times. Compare
all `locations[].mutant_id` values and statuses, checking added/missing IDs before
interpreting count changes. Historical committed records remove the checkout prefix
from IDs; remove exactly that known prefix when comparing with local raw IDs.

For the controlled timeout experiment, keep the executable and configuration fixed,
use `--workers 1`, and run three times each with `--minimum-timeout` omitted, 250,
and 1000; use 3000 if needed. Preserve the original random-order/no-abort results
separately from the selected lexical/fail-fast policy. For every changed test
selection collect a fresh raw profile and merge it before supplying
`--coverage-info`; never reuse the full-suite profile for an isolated test filter.

## CPU DAG planning audit

The public `unixlike-clang-mut` preset exposes both scopes. The DAG YAML filters
only `include/vkexec/detail/pass_execution.hpp`, with `cxx_comparison` and
`cxx_boundary`. The runtime selection is exclusively `[dag-plan]`, which constructs
no Vulkan context. Resource recording and presentation helpers in that header
remain visible as uncovered in the inclusive report. Internally, each scope uses
its own persistent instrumented executable to preserve exact source/operator
boundaries with Mull 0.35.0.

```sh
nix develop .#clang-mut
cmake --preset unixlike-clang-mut \
  -Dvkexec_MULL_WORKERS=1 -Dvkexec_MULL_MINIMUM_TIMEOUT_MS=1000
for run in 1 2 3; do
  cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_dag || exit 1
  cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_dag_uncovered || exit 1
  mkdir -p "out/dag-audit-run-$run"
  cp out/build/unixlike-clang-mut/mutation-reports/* "out/dag-audit-run-$run/"
done
ctest --test-dir out/build/unixlike-clang-mut -R 'vkexec\.mull_' --output-on-failure
```

The targets overwrite standard report names; preserve each pair before repeating.
Both variants write `vkexec_core_dag` / `vkexec_core_dag_uncovered` profiles, SQLite
and JSON under the public build's `mutation-reports/`. Keep SQLite and profiles as local
build artifacts. The [DAG baseline](mutation-dag-baseline.md) and its
[per-mutant JSON](mutation-dag-baseline.json) record initial/final results,
equivalent survivors and explicitly deferred uncovered functions.

Verify the separate range scope after changes:

```sh
cmake --preset unixlike-clang-mut
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_ranges
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_ranges_uncovered
```

The DAG scope does not change range operators or its existing 47/47 classification.
Neither audit enforces a CI mutation-score threshold.

### Resource-state audit

Use the same public preset as the ranges and DAG scopes:

```sh
nix develop .#clang-mut
cmake --preset unixlike-clang-mut
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_resource_state
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_resource_state_uncovered
```

The CPU-only `[resource-state]` selection includes deterministic allocation-fault
rollback tests when exceptions are enabled. Tests additionally tagged
`[resource-state-fault]` remain included in this selection. The tracker baseline
uses one worker, a 1000 ms minimum mutant timeout and a 10000 ms warmup timeout;
configure `-Dvkexec_MULL_MINIMUM_TIMEOUT_MS=1000` to reproduce that policy in a
fresh cache. See [the baseline](mutation-resource-state-baseline.md).
