# Resource helper and validation mutation baseline (PR 8)

This scope mutates exactly `resource_state_tracker_helpers.hpp` and
`resource_state_tracker_validation.hpp` under `include/vkexec/detail/`, with
`cxx_comparison` and `cxx_boundary`. Production code, earlier scope filters,
presets and Mull orchestration are unchanged.

| Run | Discovered | Killed | Survived | Uncovered | Timed out | Crashed / abnormal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial covered | 30 | 21 | 9 | 0 | 0 | 0 |
| Initial inclusive | 56 | 21 | 9 | 26 | 0 | 0 |
| Final covered, each of three runs | 52 | 51 | 1 | 0 | 0 | 0 |
| Final inclusive, each of three runs | 56 | 51 | 1 | 4 | 0 | 0 |

Every final covered run has the same exact ID/status map, as does every final
inclusive run. The inclusive population is the same 56 IDs from initial to final.
The mutation score is 51/52 (98.08%); inclusive evaluation coverage is 52/56
(92.86%). Uncovered mutations are neither killed nor equivalent.

| Source | Initial killed / survived / uncovered | Final killed / survived / uncovered |
| --- | ---: | ---: |
| `resource_state_tracker_helpers.hpp` | 15 / 3 / 5 | 18 / 1 / 4 |
| `resource_state_tracker_validation.hpp` | 6 / 6 / 21 | 33 / 0 / 0 |

Both sources appear in both SQLite variants and pass the existing multi-source
validator. No validator exception was added.

## Selection and behavioral assertions

The initial selection has nine tests, captured before adding new assertions. Six
direct partition/merge tests retain their old tags; three focused tracker tests
exercise initial-layout rejection, image declaration conflicts and buffer bounds.
The final selection has 21 tests, all deterministic CPU tests with fake handles.
Arguments are `[resource-helpers] --order lex --abort`; exact `--list-tests` output
is retained in the JSON metadata.

The two new helper/validation files assert the complete usage/access matrix,
every image/buffer usage's exact stage/access masks, read-write OR combinations,
explicit layout overrides, complete frontier cloning, uncovered gaps, state-aware
merging, ambiguous open-range rejection and merge idempotence. A bounded 4 x 4
model exhausts all finite rectangular cuts of a tracked image, checking every
cell and its metadata. A second fixed sequence checks newly covered holes and
updates distinct cell histories before merging.

Direct validation tests cover both image dimensions independently, exact/beyond
extent bounds, symbolic ends and unchanged rejected inputs. The start-equals-end
cases use open ends as well as finite ends so that a separate end-bound check
cannot mask a start-bound mutation. Same-pass validation enumerates access,
queue, family, step, layout, effective final layout and render-pass-transition
conditions, plus anchors and disjoint regions. Source-scope tests check exact
queue-filtered reader masks, null queue behavior and exclusion of stale anchors.
Depth/stencil tests enumerate coupled, separate, identical and disjoint masks.
The test-only allocation fault seam is also checked with scoped cleanup.

## Remaining mutations and source-emission limits

The one survivor is
`cxx_lt_to_le:include/vkexec/detail/resource_state_tracker_helpers.hpp:409:35:409:36`
(checkout prefix omitted here only). It changes `left < regions.size()` in
`merge_image_regions` to `<=`. At the extra terminal iteration, `right` starts at
`size + 1`; the inner loop is false. No state access or mutation occurs. Vector
maximum sizes prevent index overflow for realizable inputs. There is no valid
observable distinguishing input, so no production change is justified. Empty,
singleton, non-sorted and repeated-merge assertions cover the observable behavior.
The complete ID and proposed assertion rationale are in the JSON.

Four `cxx_eq_to_ne` IDs remain uncovered in `valid(image_use)` at 83:23 and 85:23,
and `valid(buffer_use)` at 105:23 and 107:23. LLVM coverage export records executed
source regions at all four locations (counts 144, 27, 189 and 27 respectively in
the diagnosed run). The runtime usage/access matrix independently requires
read-only usages to reject write/read-write, and destinations to accept only
write. Mull nevertheless does not evaluate these IDs with the indexed profile.
This is a source-coverage attribution discrepancy, not evidence that the mutations
were killed or equivalent. The local LLVM export and per-instance diagnosis are
preserved; the committed JSON retains the exact four IDs and instance counts.

All 33 emitted validation mutations are killed. Source inspection also identifies
paths with no emitted comparison/boundary mutants: switch dispatch and fallback
returns in usage/scope mapping, access-mask and source-mask bitwise OR operations,
stage/access/layout assignments, frontier handoff selection, and
`partition_image_regions` itself. The image partitioner delegates its geometric
predicates to the separately audited range algebra; its own optional checks and
range-based iteration emit no mutations in this operator set. Those paths have
exact assertions but no claim of mutation coverage. The production helpers remain
unchanged. This audit does not establish GPU synchronization correctness.

## Reproduction and evidence

```sh
nix develop .#clang-mut
cmake --preset unixlike-clang-mut -Dvkexec_MULL_WORKERS=1 -Dvkexec_MULL_MINIMUM_TIMEOUT_MS=1000
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_resource_helpers
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_resource_helpers_uncovered
```

Clang/LLVM 22.1.8, Mull 0.35.0 and Catch2 3.12.0 were used with one worker, a
1000 ms minimum timeout and a 10000 ms baseline/warmup timeout. No timeout increase
was necessary. Each variant ran serially through the existing public audit lock.
Private objects live in `mutation-builds/resource_helpers/`; independent SQLite,
JSON, text and LLVM profiles use the `vkexec_core_resource_helpers` and
`vkexec_core_resource_helpers_uncovered` report prefixes.

Raw initial reports/profiles and metadata were copied to
`out/pr8-evidence/initial/resource_helpers/` before the new assertion files were
registered. Final pairs are in `out/pr8-evidence/final/resource_helpers-{1,2,3}/`.
The [committed JSON](mutation-resource-helpers-baseline.json) preserves initial and
final reports, every mutant ID, function and source expression, source/config/test
and executable SHA-256 values, revision/dirty state, tool versions, toolchain
hashes, exact selected tests and repeated-run comparisons. Checkout prefixes are
normalized to `<checkout>` only in committed evidence. Profiles, SQLite databases,
logs and capture/comparison scripts remain in the ignored local evidence tree.
Initial tool hashes were recovered from the same immutable Nix store paths used
for those runs; initial version output was captured during the initial audit.

The earlier scopes' exact ID/status regressions, regular debug suite, formatting
and infrastructure checks are recorded in the JSON verification section.
