# Resource-state mutation baseline (PR 7)

This audit mutates only `include/vkexec/detail/resource_state_tracker.hpp` with
`cxx_comparison` and `cxx_boundary`. It extends test coverage without changing
production synchronization code, presets, or mutation orchestration. Helpers,
validation, transaction types and barrier coalescing remain outside this scope.
The historical DAG survivor classifications are unchanged.

| Measurement | Discovered | Killed | Survived | Uncovered | Timeout/crash/abnormal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial covered | 50 | 44 | 6 | 0 | 0 |
| Initial inclusive | 56 | 44 | 6 | 6 | 0 |
| Final covered (each of three runs) | 56 | 54 | 2 | 0 | 0 |
| Final inclusive (each of three runs) | 56 | 54 | 2 | 0 | 0 |

The inclusive population stays at the same 56 exact IDs from initial to final.
Each final variant has identical complete ID/status mappings across three runs.
The final mutation score is 54/(54+2), or 96.43%; evaluation coverage is 56/56,
or 100%. Neither statistic counts uncovered mutants as killed or equivalent.
No score gate is introduced.

## Evidence and reproduction

Use the existing single public preset:

```sh
nix develop .#clang-mut
cmake --preset unixlike-clang-mut -Dvkexec_MULL_WORKERS=1 -Dvkexec_MULL_MINIMUM_TIMEOUT_MS=1000
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_resource_state
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_resource_state_uncovered
```

The private instrumented tree is `out/build/unixlike-clang-mut/mutation-builds/resource_state/`.
Public reports remain in `out/build/unixlike-clang-mut/mutation-reports/` with
`vkexec_core_resource_state` and `vkexec_core_resource_state_uncovered` prefixes.
No resource-state-specific infrastructure changes were necessary.

The initial evidence was captured after registration and tagging, before adding
assertions. Local SQLite, JSON, text, coverage profiles, metadata and selected
names remain under `out/pr7-evidence/initial/`; final pairs are preserved separately
under `out/pr7-evidence/final-{1,2,3}/`. `comparison.json` compares exact local IDs
and statuses. The committed [JSON evidence](mutation-resource-state-baseline.json)
retains both initial and final reports, all mutant IDs, source expressions,
function classifications, source/configuration/test/binary SHA-256 fingerprints,
source revision, tool versions and exact selected test names. Checkout prefixes
are normalized to `<checkout>` in the committed evidence only.

The toolchain is Clang/LLVM 22.1.8, Mull 0.35.0 and Catch2 3.12.0. The selection
is `[resource-state] --order lex --abort`, with one worker, a 1000 ms minimum
mutant timeout and a 10000 ms warmup/baseline timeout. The initial warmup was
about 23 ms; complete mutant evaluation took about 1.1 seconds. No timeout
classification appeared in initial or repeated final runs, so no timeout increase
was needed. Audits ran serially through the existing public lock.

The initial selection contains 61 tests; the final selection contains 69. All
are deterministic CPU tests with fake handles. Direct tracker tests in the
buffer/image region suites retain their previous `[ranges]` and `[pass]` tags.
Pure range algebra, generic DAG planning, direct dependency injection and GPU
submission tests are excluded. The exception-enabled build includes allocation
fault tests; the new `[resource-state-fault]` test is included by the main tag.
Scoped cleanup disarms fault injection during assertion unwinding.

## Behavioral invariants

The focused test source adds eight cases, complementing the existing byte and
subresource reference-model tests:

- All seven requested buffer read/write sequences assert exact dependency edges,
  barrier byte ranges, stage/access masks and ignored queue-family indices.
- Exclusive/concurrent sharing, same/different queues and same/different families
  check ordering and complete release/acquire masks and image layouts.
- Extent re-registration checks resource identity, dimensions, sharing mode and
  inconsistent extents; invalid step indexes include the exact end boundary.
- Failed buffer and image batches compare every dependency, barrier field,
  image-use record and region count against a fresh tracker after a valid retry.
  New resource rollback is checked by registering different extents afterward.
- Repeated modifications of pre-existing resources exercise earliest journal
  checkpoints. Deterministic allocation failures cover extent creation and
  initial image barrier generation, followed by full-plan comparison after retry.
- Attachment initial layouts include UNDEFINED and GENERAL, with explicit final
  shader-read layout. Subsequent reads check effective old/new layouts and masks.

Existing tagged tests cover disjoint/adjacent/overlapping buffer regions, reader
fan-in and ownership handoff, mip/layer independence, partial image intersections,
whole-resource accesses, exact/beyond-extent ranges, imported layouts, compatible
reads, read-only layout changes and mixed depth/stencil rejection. Their detailed
range/barrier assertions are retained.

## Function-level findings

Ten initially non-killed mutations now fail relevant assertions: three invalid
input/attachment survivors, one image journal survivor, four image extent
mutations and two journal predicates that were initially uncovered. The committed JSON contains the exact
expression and classification of every discovered mutation.

Dependency/frontier generation, buffer and image state advancement, ownership
handoff and extent registration have no remaining surviving or uncovered emitted
mutations. Rollback has the following precise limitations:

| Location / mutation | Outcome | Reason and distinguishing input |
| --- | --- | --- |
| `remember_buffer`, 69:36, `cxx_ge_to_gt` | Survived | For a batch starting with N buffers, reuse newly created index N. The mutant saves an unnecessary checkpoint; commit discards it and rollback restores then removes it by resizing to N. Equivalent for plan outcomes when allocations succeed. Extra copies can allocate, so this is not a claim of equivalence under arbitrary allocator failure. |
| `remember_image`, 80:36, `cxx_ge_to_gt` | Survived | Same reasoning for the first image created in a batch. |

The initial uncovered journal predicates at 71:99 and 82:98 (`cxx_eq_to_ne`)
needed two distinct pre-existing resources of each kind in the same transaction.
A duplicate checkpoint can still restore the earliest snapshot in reverse order,
so repeating only one resource was insufficient. The distinct-resource regression
requires separate checkpoints and compares the complete retried plan with a fresh
tracker. Both mutants now demonstrate test failures and are classified killed.
The initial uncovered image extent predicates at 469:97, 475:52, 476:40 and
476:94 are killed by consistent/inconsistent registration tests. No uncovered
mutants remain. Final LLVM coverage diagnosis reports all 56 locations source
covered; raw evidence is `out/pr7-evidence/coverage-diagnosis.json`.

These operator families do not generate mutations for every rollback resize or
barrier assignment. Passing this audit therefore does not prove every possible
rollback defect is detectable. Whole-plan comparisons add independent protection;
GPU submission and runtime synchronization validation remain deferred.

## Regression verification

The existing range audit remains 47 killed, zero survivors and zero uncovered.
The DAG audit remains 36 killed, two survivors and 19 inclusive uncovered; its
historical survivor classifications are unchanged. All three `vkexec.mull_`
infrastructure CTest checks pass. The regular debug suite reports 362 passed,
three GPU-dependent skips and no failures (365 total). Changed C++ files pass
`clang-format --dry-run --Werror`; CMake files pass `cmake-format --check` and the
repository line-length check passes. The new test source is 472 lines, below the
800-line test limit. Logs are retained under `out/pr7-evidence/`.
