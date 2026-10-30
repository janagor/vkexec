# Mutation audit scopes

The single public `unixlike-clang-mut` preset exposes every registered audit.
Mull generates mutations during compilation. A real Mull 0.35.0 probe showed that
runtime path filtering does not isolate uncovered-inclusive reports, and runtime
operator lists do not narrow the emitted operators. Consequently, audit targets
orchestrate persistent, isolated internal builds rather than sharing an
instrumented executable. See the [runtime-filter experiment and preserved
baselines](mutation-dag-baseline.md#single-preset-fallback).

Register scopes after their executable targets exist and before the single
`vkexec_finalize_mutation_scopes()` finalization call:

```cmake
vkexec_register_mutation_scope(
  NAME ranges
  CONFIG "${PROJECT_SOURCE_DIR}/mull.yml"
  TEST_TARGET vkexec_core_tests
  SOURCES "include/vkexec/detail/resource_ranges.hpp"
  TEST_ARGS "[ranges]" "--order" "lex" "--abort")
```

Every field is required. Names use lowercase letters, digits and underscores,
starting with a letter, and must be unique. CONFIG must name an existing YAML
file; TEST_TARGET must be an existing executable. SOURCES is a nonempty list of
exact repository-relative files, without duplicates or paths outside the root.
TEST_ARGS is a nonempty CMake argument list, passed identically to baseline
profiling and Mull. Keep arguments separate; do not pass a shell command.

Finalization copies each normalized scope YAML to `mutation-configs/<name>.yml`
and creates both `vkexec_mutation_<name>` and `vkexec_mutation_<name>_uncovered`.
There is no public active-scope setting. An old cached `vkexec_MULL_SCOPE` is
removed on reconfiguration. The public build does not instrument its own targets.

`cmake/RunMutationScope.cmake` configures the selected private tree under
`mutation-builds/<name>/` using a generated initial-cache file. Toolchain settings,
project options, worker/timeout values and downloaded dependency sources are
forwarded; object files and dependency build directories stay isolated. Changes to
these settings are applied on the next audit invocation. Internal build parallelism
uses `CMAKE_BUILD_PARALLEL_LEVEL`, defaulting to two jobs to bound memory use.
`vkexec_MULL_INTERNAL_SCOPE` and `vkexec_MULL_REPORT_DIR` are private orchestration
settings, not switches users need to set.

Within each private tree, `vkexec` and the selected test executable are instrumented
once. The scope YAML is copied to its `mull.yml` for compilation and
`mutation-configs/<name>.yml` for the runner; the contents are identical. Changes to
the compile configuration invalidate instrumented objects. Compiler launchers
bypass caching. Running a different public audit uses its existing private tree,
without replacing or rebuilding the first scope's executable.

Both variants remove stale reports/profiles, collect fresh matching-filter LLVM
coverage, merge the indexed profile, execute Mull and validate expected source
paths. All reports retain their existing names in the public build's
`mutation-reports/vkexec_core_<name>*`, including separate covered/uncovered raw
profiles, indexed profiles, SQLite, JSON and text output. Missing or unexpected
source files fail validation. Strict validation requires killed or surviving
mutants; entirely uncovered reports remain parseable for diagnostics.

A process lock serializes the public audit commands, including commands from
separate build processes, to avoid concurrent mutation timing and shared-tree
build races. Configure, compilation, baseline, merge, runner and validator failures
propagate to the public target. There is no aggregate target or CI score gate.

The DAG registration uses `cmake/mull/dag.yml`,
`include/vkexec/detail/pass_execution.hpp` and `[dag-plan]`. Its two equivalent
survivors and intentionally uncovered helpers remain visible. Range operators,
report names and 47/47 classification are preserved. Independent scores are never
combined.

Run the focused configure checks without Mull or Vulkan:

```sh
python3 test/mutation/audit_registration/test_registration.py
```

## Recovering persistent internal builds

After a generator or toolchain change, a private CMake cache may no longer match
the public build. Stop audit processes and preserve any reports you need before
cleaning generated build trees. For an affected private scope, remove its tree
and rerun the public target; the target configures and rebuilds it automatically:

```sh
cmake -E rm -rf out/build/unixlike-clang-mut/mutation-builds/dag
cmake --preset unixlike-clang-mut
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_dag
```

Use the corresponding `ranges` directory for the range audit, or remove both
private scope directories when changing toolchains. If the public build's
generator changes, recreate the public build directory too, after copying reports
outside it. Re-enter the matching development shell before configuring.

Cache forwarding deliberately covers project options, the supported toolchain
settings and dependency source hints. When adding options that affect audit
builds, review `_vkexec_write_mutation_cache()` and extend its focused tests as
needed; do not forward every parent cache entry, including build-directory state.

## PR 5 verification

The saved pre-refactor report pair and SHA-256 manifest are under
`out/pr5-evidence/before/`. Three covered and three uncovered audits each
produced 47 discovered and 47 killed mutants, zero survivors, timeouts or
uncovered mutants, and 100% mutation score and evaluation coverage. Each run's
complete mutant ID/status set equals the saved report for that variant, without
requiring checkout-root normalization. Reports, profiles and logs are preserved
under `out/pr5-evidence/{ranges,ranges_uncovered}-{1,2,3}/`; the comparison is
`out/pr5-evidence/comparison.json`. These local generated artifacts are not
committed. Historical PR 4 records remain unchanged.

All three PR 3 standalone binaries passed `--verify-fixture`. The out-of-line
indexed report validated against both explicit source paths: ten discovered,
six killed, two survived and two uncovered. Its summary is
`/tmp/mull-multifile-validation.json`.

The mutation CTest checks pass, including focused configure and stub execution
checks for argument forwarding and failures in baseline execution, profile merge,
Mull execution and validation. The normal debug CTest suite completed without
failures: 348 passed and three skipped on this machine.

## Resource-state tracker

The `resource_state` scope uses `cmake/mull/resource_state.yml`, mutates only
`include/vkexec/detail/resource_state_tracker.hpp`, and selects CPU-only
`[resource-state]` tests with lexicographic ordering and fail-fast execution.
Helpers, validation, checkpoint types and barrier coalescing support those tests
but are not mutation sources. The private build is
`mutation-builds/resource_state/`; covered and inclusive reports use the distinct
`vkexec_core_resource_state` and `vkexec_core_resource_state_uncovered` prefixes.
See [the tracker baseline](mutation-resource-state-baseline.md) for evidence and
remaining limitations. This scope adds neither a preset nor a score gate.

## Resource helpers and validation

The `resource_helpers` scope selects `[resource-helpers] --order lex --abort`
and uses `cmake/mull/resource_helpers.yml`. Its two expected mutation sources are
`include/vkexec/detail/resource_state_tracker_helpers.hpp` and
`include/vkexec/detail/resource_state_tracker_validation.hpp`. Both must appear in
the SQLite report; the existing multi-source validator enforces this requirement.
The focused CPU tests exercise usage mapping, region/frontier preservation,
extent bounds, declaration conflicts and queue-filtered source scopes.

## Barrier coalescing

The independent `barrier_coalescing` scope selects
`[barrier-coalescing] --order lex --abort`, uses
`cmake/mull/barrier_coalescing.yml` and mutates only
`include/vkexec/detail/resource_barrier_coalescing.hpp`. Exact merge assertions
and bounded byte/subresource reference models check synchronization metadata,
coverage, ownership-transfer preservation and fixed-point behavior.

Both scopes use `cxx_comparison` and `cxx_boundary`, the same single public preset,
and private `mutation-builds/resource_helpers/` and
`mutation-builds/barrier_coalescing/` directories. Report prefixes are
`vkexec_core_resource_helpers` and `vkexec_core_barrier_coalescing`, each with a
separate `_uncovered` variant. Their scores are independent of the three earlier
audits. See [helper evidence](mutation-resource-helpers-baseline.md) and
[coalescing evidence](mutation-barrier-coalescing-baseline.md).

## CI baseline review

Portable snapshots in `mutation-ci-baselines/` use CI schema 1, distinct from
the validator's schema 2 reports. They retain the complete inclusive population
and normalized IDs with source locations. Historical evidence remains unchanged;
a historical dirty measurement must never be labeled as a clean commit result.

Baseline creation is an explicit operation, separate from ordinary CI. Use a
clean checkout of the revision being reviewed, including committed test changes.
The baseline writer and audit runner may be invoked by absolute path from an
external checkout when reproducing an older revision. Do not copy scripts into
the measured checkout, which would change its clean state. Run:

```sh
nix develop .#clang-mut --command bash scripts/ci/run_mutation_audits.sh all \
  --measure-only --artifacts out/mutation-baseline-evidence
python3 scripts/create_mutation_baseline.py --scope dag --repository-root . \
  --covered out/mutation-baseline-evidence/dag/covered.json \
  --inclusive out/mutation-baseline-evidence/dag/inclusive.json \
  --manifest out/mutation-baseline-evidence/manifest.json \
  --output /tmp/reviewed-dag-baseline.json
```

Repeat the writer for each measured scope, then review and commit snapshots.
The writer requires a clean current revision, clean-run manifest, successful
covered and inclusive targets, matching report hashes, matching measurement
fingerprint, and consistent per-mutant outcomes. It never accepts historical
wrapper documents as report inputs. Save the run artifacts with the review.

Review exact IDs, status transitions and separate fingerprints. Explain source
or operator population changes and any timeout/classification changes. An
improvement elsewhere cannot offset a scope regression. Never regenerate a
baseline automatically to make CI pass. CI starts in report mode; switch the
workflow's explicit policy to `regression` only after repeated clean hosted runs
reproduce the committed baseline. Hosting a workflow and recording a successful
manual GitHub dispatch remain separate from local verification.

The CI runner uses a 1000 ms minimum mutant timeout and a 10000 ms warmup timeout
for all scopes, with one worker. The original range experiment used 250 ms;
review the clean CI reproduction evidence before accepting that timeout change.
No CMake timeout extension is needed when classifications agree.
