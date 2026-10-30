# DAG planning mutation baseline

PR 6 audits CPU planning in `include/vkexec/detail/pass_execution.hpp` using
`cmake/mull/dag.yml`, independently of the resource-range audit. The selected
operators are `cxx_comparison` and `cxx_boundary`. `[dag-plan] --order lex --abort`
selects only tests that use fake queue handles and never construct a Vulkan context.
The filter applies at runtime; the YAML selects the entire header at compilation.

## Reproduction and provenance

The source revision was `555a02dcbae7e3e8153de6c122743b25acdd3d95`, with the
uncommitted PR 6 changes identified by file hashes in the
[JSON record](mutation-dag-baseline.json). That record preserves the production
header, YAML, test and build-file hashes, initial/final binary hashes, exact test
names and arguments, and every discovered mutant ID and status. Clang and LLVM
were 22.1.8, Mull 0.35.0 and Catch2 3.12.0, using the existing `clang-mut`
environment. No production implementation changed.

```sh
nix develop .#clang-mut
cmake --preset unixlike-clang-mut \
  -Dvkexec_MULL_WORKERS=1 -Dvkexec_MULL_MINIMUM_TIMEOUT_MS=1000
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_dag
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_dag_uncovered
ctest --test-dir out/build/unixlike-clang-mut -R 'vkexec\.mull_' --output-on-failure
```

Mull's process tracing requires execution outside this workspace's sandbox. The
initial sandboxed runner aborted; it did not produce an accepted baseline.
Dependency downloads and the existing debug ccache also needed sandbox permission.
The successful runs used the same toolchain and unchanged audit infrastructure.

The commands above describe the current single-preset interface. The original
initial/final measurements below were collected before the build-layout refactor,
in separately selected build directories. Those records and hashes are retained;
the single-preset regression is reported separately below.

## Initial and final measurements

The initial pair was collected with only the two existing tests retagged, before
adding the new source to the build or changing assertions. Its reports and hashes
were saved before strengthening tests.

| Report | Discovered in report | Killed | Survived | Timed out | Crashed / abnormal | Not covered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial covered-only | 13 | 11 | 2 | 0 | 0 / 0 | 0 |
| Initial uncovered-inclusive | 57 | 11 | 2 | 0 | 0 / 0 | 44 |
| Final covered-only, each of 3 runs | 38 | 36 | 2 | 0 | 0 / 0 | 0 |
| Final uncovered-inclusive, each of 3 runs | 57 | 36 | 2 | 0 | 0 / 0 | 19 |

All 57 discovered IDs in the inclusive baseline remain identical after adding
assertions. Covered-only reports omit uncovered mutants; they are not a separate
universe of discovered implementation mutations. The raw final mutation score is
36/38 (94.74%), without removing equivalent survivors from its denominator.
The inclusive evaluation coverage is 38/57 (66.67%), a separate diagnostic.

All three final covered-only ID/status lists match exactly, as do all three
inclusive lists. The instrumented binary hash is identical across these runs.
Reports and profiles are preserved separately in
`out/pr6-evidence/{initial,strengthened,final-1,final-2,final-3}/`; they are local
build artifacts. The JSON records per-run artifact hashes and equality checks.
Opaque IDs include this checkout's absolute prefix; only that prefix should be
normalized for a different checkout.

One worker, lexical ordering, fail-fast assertions, a 1000 ms minimum mutant
timeout and the existing 10000 ms runner timeout were retained throughout. Final
covered/inclusive runner totals were respectively 343/333, 328/338 and 324/327 ms;
these are whole runner timings, not per-mutant timeout recommendations. No accepted
run timed out, crashed or exited abnormally. There was no timeout inflation.

## Tests and resolved survivors

Six new CPU-only cases in `dag_planning_tests.cpp` cover:

- Empty, single-root, independent, same-queue, chain and diamond DAGs, with exact
  step bounds, queue handles/families and incoming edges; mismatched sizes, self,
  future, duplicate and out-of-range predecessors are rejected.
- Empty, chain, independent, diamond and disconnected terminal sets in index order.
- Valid and invalid contiguous partitions, empty plans, zero-length/reversed
  ranges, gaps, overlaps, incomplete/overlong coverage and predecessor ordering.
- Five resource steps partitioned as `[0,2)`, `[2,4)`, `[4,5)`, producing exactly
  `[]`, `[0]`, `[1,0]`; empty/same-batch hazards, repeated batch pairs, existing
  edges and multiple source batches are checked independently.
- Default sequential affinities, explicit handles and families, adjacent queue
  grouping, exact sequential predecessors and invalid queue counts.
- Empty/single/same-family/cross-family plans for `crosses_queue_families()`.

The initial survivors were precisely:

```text
cxx_gt_to_ge:/home/janagorski/Projects/currently/vkexec/include/vkexec/detail/pass_execution.hpp:34:31:34:32
cxx_gt_to_le:/home/janagorski/Projects/currently/vkexec/include/vkexec/detail/pass_execution.hpp:34:31:34:32
```

Both mutate `plan.batches.size() > std::size_t{1}` in `plan_batches()`.
The old tests checked ranges and queues but did not assert sequential incoming
edges. The new test requires `[]`, `[0]`, `[1]`, detecting both an erroneous
predecessor on the first batch and missing edges on later batches. Both mutants
are killed in every final run and the original implementation passes.

## Remaining equivalent survivors

```text
cxx_lt_to_le:/home/janagorski/Projects/currently/vkexec/include/vkexec/detail/pass_execution.hpp:310:48:310:49
cxx_lt_to_le:/home/janagorski/Projects/currently/vkexec/include/vkexec/detail/pass_execution.hpp:311:58:311:59
```

These change `source < batch.end_step` and `destination < batch.end_step` to
`<=` in `add_resource_dependencies()`. On valid, contiguous, nonempty partitions,
a valid step at a batch's end is the first step of the next batch. The mutated
comparison temporarily assigns it to the previous batch, then the ascending scan
overwrites that mapping with the correct next batch. All other valid indexes have
the same mapping. The last end equals `step_count`, which is outside the valid
dependency-index domain. Thus both mutants are equivalent on the tested contract;
no assertion can distinguish them within that domain. They remain visible as
survivors. No invalid-index contract or production optimization was introduced.

The repository currently uses a scan over batches in this helper, rather than
the `step_to_batch` lookup described in the proposal. This PR preserves that
implementation, including predecessor insertion order.

## Function attribution and deferred coverage

| Function | Killed | Survived | Not covered |
| --- | ---: | ---: | ---: |
| `same_queue` | 2 | 0 | 0 |
| `plan_batches` | 2 | 0 | 0 |
| `plan_dag` | 6 | 0 | 0 |
| `plan_execution` | 1 | 0 | 0 |
| `validate_execution_batches` | 11 | 0 | 0 |
| `crosses_queue_families` | 3 | 0 | 0 |
| `terminal_batches` | 2 | 0 | 0 |
| `add_resource_dependencies` | 9 | 2 | 0 |
| `declare_indexed_step` | 0 | 0 | 1 |
| `plan_resource_sync` | 0 | 0 | 2 |
| `record_sync_batch` | 0 | 0 | 3 |
| `validate_presentation_image` | 0 | 0 | 13 |

The 19 uncovered mutations belong to resource-declaration dispatch (line 212),
per-step resource synchronization planning (line 227), command recording (lines
253 and 264), and presentation-image validation (lines 280–294). The first two
are CPU resource-state/synchronization helpers deferred to that separate audit;
the latter two concern recording/presentation behavior outside this CPU DAG
selection. They are not uncovered mutations of DAG partitioning, graph validation,
terminal detection or resource-edge injection. Their exact IDs, source lines,
owning functions and statuses remain in the JSON. File filtering does not guarantee
that every expression or template instantiation produces a mutant, so this result
is limited to the 57 emitted mutations and these two operator families.

## Validation

The normal Clang debug CPU selection passes 298 assertions across 8 cases without
constructing a Vulkan context. All three mutation diagnostic CTests pass. The
normal `vkexec_tests` build succeeds and the complete debug CTest suite has 354
passes, 3 capability-dependent skips and no failures (357 total).

Both independent range targets pass with 47 discovered, 47 killed and zero other
outcomes. Their exact mutant ID/status sets equal the saved PR 5 pre-refactor
reports in `out/pr5-evidence/before/mutation-reports/`. The regression reports are
preserved in `out/pr6-evidence/ranges/` and their JSON hashes are recorded in the
baseline JSON. `cmake/MutationTesting.cmake` and `scripts/validate_mull_report.py`
are unchanged. No GPU implementation, batching optimization or CI score gate was
added.


## Single-preset fallback

The revised public workflow uses one preset, without changing an active-scope
cache variable:

```sh
nix develop .#clang-mut
cmake --preset unixlike-clang-mut
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_ranges
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_dag
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_ranges_uncovered
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_dag_uncovered
```

A small real Mull 0.35.0 / LLVM 22.1.8 probe was run before choosing the build
layout. Two headers each contain a comparison and arithmetic operations. One
executable was compiled with their union of paths and the existing three operator
families. Each audit then supplied a narrower path YAML, its own test argument,
and a freshly collected indexed profile. The DAG runner YAML enabled only the two
DAG operator families. Both covered-only and uncovered-inclusive runs used that
unchanged binary.

| Probe report | Expected isolated count | Observed count | Observation |
| --- | ---: | ---: | --- |
| Range covered | 4 | 4 | Covered path filtering works |
| Range inclusive | 4 | 8 | Four uncovered DAG mutants leak into the range report |
| DAG covered | 2 | 4 | Compiled arithmetic mutants still execute |
| DAG inclusive | 2 | 8 | Both extra operators and uncovered range mutants remain |

A second diagnostic used an include regex that cannot match any source; covered
mutants were removed, but previously classified uncovered mutants remained in the
report. The release's filter-chain source applies coverage before path filtering
and removes `ReportAs` mutants from subsequent filters. The runner does not apply
a mutator-selection filter to already emitted mutations. This explains the probe
results and makes a union binary unsuitable for the required exact reports. The
[filter reference](https://mull-project.com/reference/filters/) describes path
selection, while the [setup guidance](https://mull-project.com/getting-started/setup/)
recommends matching generation and execution configs; the local measurements are
the acceptance evidence for this pinned binary.

The complete probe, reproduction script, executable, YAMLs, SQLite reports, profiles
and logs are retained locally in `out/pr6-evidence/runtime-filter-probe/`. Its
per-mutant outcomes and hashes are included in the baseline JSON. No unexpected
source entries were deleted from reports to make the experiment pass.

The authorized fallback keeps persistent instrumented builds under
`out/build/unixlike-clang-mut/mutation-builds/{ranges,dag}/`. Public targets configure
and build those trees automatically. Each scope uses identical compiler/runner
configuration contents, applies instrumentation once per selected target, and
writes independently named reports and fresh profiles into the public build's
`mutation-reports/`. Scope switching does not replace either binary. A process
lock serializes public audits. Internal compilation defaults to two jobs, with
`CMAKE_BUILD_PARALLEL_LEVEL` available to override it. There is no second public
mutation preset, combined production YAML, aggregate target or score gate.

The original initial and three final DAG measurements above are preserved as
historical evidence. The following single-preset regression results are recorded
separately in the JSON rather than replacing those measurements.


All four public targets were run consecutively three times after both private
builds were populated. Every range ID/status list equals the original 47-mutant
baseline, and every DAG inclusive list equals the original 57-mutant final
baseline (36 killed, 2 equivalent survivors, 19 uncovered). Covered DAG reports
retain the same 38 IDs/statuses. Validation rejects unexpected source files.
Each invocation generated new raw and indexed profiles with its matching Catch2
filter. Across the entire sequence, both private executables retained identical
hashes and modification times, and the logs contain no compile or link commands.
This verifies reuse of the two persistent binaries; a single shared binary was
explicitly rejected by the probe.

Reports are preserved individually under
`out/pr6-evidence/single-preset/<audit>-<run>/` for runs 1–3. Its `comparison.json`
and the committed JSON's `single_preset_refactor` preserve all comparisons, hashes,
profile-freshness checks and probe outcomes; the probe's complete fixture sources
and reproduction script are embedded in that JSON. The original pre-refactor
measurements remain intact in the original JSON fields and evidence directories.

All three public-build mutation diagnostic CTests pass, including updated
registration tests for both scopes, per-scope arguments/configurations, one-time
private instrumentation, relative paths and failure propagation. The cache seed
also round-trips paths with spaces and values containing quotes, semicolons,
bracket terminators and literal dollar expressions. The normal debug build still
succeeds, with 354 passing tests, 3 capability-dependent skips and no failures.
The DAG tests, scope YAML, resource-range YAML and production header are unchanged
from the accepted pre-refactor measurements.


## Finalized implementation revision

The finalized implementation revision is `fe00606e30b60c05b10e60ecca7213d3672c7531`
(`refactor(mutation): expose isolated audits through one preset`), which includes
DAG test commit `5097de8`. Every source-file hash recorded in
`single_preset_refactor.source_sha256` matches that committed implementation.

The source revisions and dirty-tree hashes in the measurement records still
identify the original historical snapshots. Those measurements have not been
rewritten or relabeled as runs from the committed revision. The separate
`finalized_implementation` JSON entry records the commit and this source-hash
verification.
