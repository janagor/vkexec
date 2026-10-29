# Mutation testing

Mull audits whether the existing Catch2 tests detect deliberate implementation changes. Keep unit tests, coverage, sanitizer runs, and Vulkan integration tests alongside this audit.

The opt-in `unixlike-clang-mut` preset instruments the compiled core library and the core test executable, including header-defined implementation code. The initial `mull.yml` selects only `include/vkexec/detail/resource_ranges.hpp`; tests and fetched dependencies are excluded. The `vkexec_mutation_ranges` target selects the device-independent `[ranges]` tests in `vkexec_core_tests`.

## Toolchain and execution

On x86_64-linux and aarch64-linux, the `clang-mut` shell provides Mull 0.35.0 for LLVM 22.1.8. Only `clang-mut` includes mutation tools and exports their paths; the default, `clang`, and `clang16` shells do not include Mull:

```sh
nix develop .#clang-mut
cmake --preset unixlike-clang-mut
```

The `unixlike-clang-mut` preset inherits the normal Clang debug toolchain, reads `MULL_PLUGIN` and `MULL_RUNNER` from the environment, and enables mutation testing. It disables hardening, IPO, coverage tooling, sanitizers, static analyzers, optional modules, and compiler caching for this narrow audit. Ordinary debug and release presets retain their original settings and do not enable Mull. If an existing build directory has mutation testing cached as enabled, configure it once with `-Dvkexec_ENABLE_MUTATION_TESTING=OFF` to clear that setting.

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

Record these versions and confirm that `clang++` is the compiler selected in the CMake cache. A matching major-version suffix and an existing plugin file do not prove LLVM build compatibility. The instrumented compilation must successfully load the supplied plugin; inspect the verbose build (`cmake --build out/build/unixlike-clang-mut --target vkexec_core_tests --verbose`) and Mull's configuration messages before accepting the toolchain. Plugin loading has been verified by recompiling both selected targets with this frontend and running their generated mutants.

The [baseline record](mutation-baseline.md) contains measured results and their limitations. Building this target compiles the instrumented tests and executes Mull. It is separate from ordinary CTest and is never part of the default build.

The preset disables IPO, hardening, sanitizers, coverage tooling, compiler caching, static analyzers, examples, and optional modules. Mull uses its own LLVM profiling flags for covered-code selection. Instrumentation is private to the two selected targets; third-party targets are not instrumented. CMake copies `mull.yml` to the build root and explicitly sets `MULL_CONFIG` for both instrumented compiler commands and the runner. A configuration digest changes the selected targets' compile definitions when the YAML changes, forcing mutant regeneration after CMake reconfiguration. Verify that both compiler and runner diagnostics name the copied build-directory configuration on the first real run.

The warmup/baseline timeout defaults to 10000 milliseconds, adjustable with `-Dvkexec_MULL_TIMEOUT_MS=...`. Mull 0.35 derives mutant timeouts from ten times the measured baseline duration, with a default minimum of 30 ms; `--timeout` does not set a fixed per-mutant budget. Keep timeout outcomes separate when comparing scores.

## Review the baseline

Mull prints findings and writes `out/build/unixlike-clang-mut/mutation-reports/vkexec_core_ranges.sqlite`. Each target removes its previous reports and profiles, runs the unmodified `[ranges]` tests with a fresh per-audit `LLVM_PROFILE_FILE`, merges that raw profile with matching `llvm-profdata`, runs Mull with `--coverage-info` pointing to the indexed profile, and validates the report using Python's standard-library SQLite support. The `.profraw` and `.profdata` files remain beside the reports for inspection. This works around the [verified multi-TU raw-profile attribution limitation](mutation-coverage-attribution.md); both compiled targets retain the mutation pass and coverage instrumentation. Surviving mutants are allowed during baseline collection; build, original-test, runner, or report-validation errors fail the target.

The validator requires a report containing only the expected source path and at least one killed or surviving mutant. It reports killed, survived, timed-out, crashed, abnormal-exit, and uncovered counts separately. The mutation score is `100 * killed / (killed + survived)`; timeouts, crashes, abnormal exits and uncovered mutants are excluded from this denominator, never counted as kills. Reports containing only these excluded outcomes fail validation. Missing/empty reports, unknown statuses, dry-run results, and unsupported schemas fail validation. Its schema and status definitions match locally generated Mull 0.35.0 reports and were inspected in [Mull's upstream source](https://github.com/mull-project/mull/blob/dab0b590af1ae1ec34609d993668a9a12a703c30/rust/mull-reporters/src/sqlite_reporter.rs), with killed and uncovered statuses observed in real runs. Other execution statuses remain covered by validator fixtures.

Inspect killed and surviving mutants with the matching reporter:

```sh
MULL_CONFIG="$PWD/out/build/unixlike-clang-mut/mull.yml" mull-reporter-22 \
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

For later targets, keep separate SQLite report names and aggregate them using the matching `mull-reporter`. Expand to DAG planning and resource-state tracking before GPU execution or concurrency stress tests. Introduce scheduled CI, diff-based PR runs, and score regression thresholds after the toolchain and baseline have been validated; this initial integration does not enable a CI gate.

See Mull's [setup guide](https://mull-project.com/getting-started/setup/), [runner reference](https://mull-project.com/reference/mull-runner/), and [filters](https://mull-project.com/reference/filters/) for tool behavior and reporting options.

## Investigating uncovered locations

See the [source-coverage investigation](mutation-coverage-investigation.md) for the measured source-region discrepancy and reproduction commands. Do not infer missing assertions from Mull uncovered status alone.

The [coverage-attribution investigation](mutation-coverage-attribution.md) records automatic/raw/indexed comparisons, isolated instrumentation controls, the standalone reproduction, and the corrected baseline. The initial 4/47 audit remains experimental historical evidence. Indexed profiles evaluate all 47 discovered mutants; surviving and timed-out mutants remain visible, with no CI score gate.
