# Barrier coalescing mutation baseline (PR 8)

This independent scope mutates only
`include/vkexec/detail/resource_barrier_coalescing.hpp`, with `cxx_comparison` and
`cxx_boundary`. It shares the existing public preset and orchestration, with its
own private executable, SQLite databases and LLVM profiles.

| Run | Discovered | Killed | Survived | Uncovered | Timed out | Crashed / abnormal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial covered | 12 | 12 | 0 | 0 | 0 | 0 |
| Initial inclusive | 16 | 12 | 0 | 4 | 0 | 0 |
| Final covered, each of three runs | 16 | 15 | 1 | 0 | 0 | 0 |
| Final inclusive, each of three runs | 16 | 15 | 1 | 0 | 0 | 0 |

All final covered/inclusive variants reproduce the exact ID/status map across
three runs. All 16 initial inclusive IDs remain present. The final mutation score
is 15/16 (93.75%); evaluation coverage is 16/16. Neither number proves that every
source expression generated a mutant. No score gate is introduced.

## Assertions and mutation findings

The initial selection contains the two existing direct coalescing cases, with
all previous tags retained. The final selection contains six CPU-only cases with
`[barrier-coalescing] --order lex --abort`. Complete selected names are in the
JSON metadata. The new source checks every synchronization-scope equality field
independently for both barrier types, resource identity, queue families, ownership
transfers, invalid/empty ranges and unchanged left operands on rejected merges.
Buffer merges verify exact offset/size in both adjacency orders. Image merges
verify every field of `VkImageSubresourceRange`, both adjacency dimensions,
layouts, aspects and rejection of nonrectangular unions.

Fixed-point tests cover empty and singleton inputs, two adjacent barriers, three
adjacent buffer pieces in non-sorted/interleaved order, incompatible metadata
groups, distinct ownership-transfer ranges and idempotence. A non-sorted 4 x 4
image grid requires repeated merging. Independent byte/cell maps compare resource
identity, coverage, aspect, stages/accesses, layouts and family metadata before
and after coalescing. The reference domain forbids overlapping input coverage;
duplicate or expanded cells fail insertion/count/equality assertions. Ownership
transfers retain their individual operations and exact ranges.

The one survivor is
`cxx_lt_to_le:include/vkexec/detail/resource_barrier_coalescing.hpp:55:35:55:36`
(checkout prefix omitted here only), changing the outer `left < barriers.size()`
to `<=`. The extra final iteration starts `right` at `size + 1`; its inner loop
never executes. No observable state changes, and realizable vector sizes prevent
index overflow. There is no valid distinguishing input. The committed JSON records
the exact ID, expression, function and regression-assertion rationale.

An intermediate test version evaluated the opposite outer-loop mutation as a
timeout because the first input was empty, causing an endless scan. Putting a
mergeable input before the empty case exposes the missing merge by assertion and
kills that mutation without changing the timeout policy. Final outcomes contain no timeout or crash. No production defect or
production edit was required.

Source inspection shows comparison mutations in scope equality, merge guards and
loop bounds. Range/layout assignments and loop-control boolean assignments do not
generate mutations with these operators. Underlying range normalization and
rectangle algebra belong to the existing range audit, not this source population.
The new reference models complement those audits without claiming complete
statement or GPU synchronization coverage.

## Reproduction and evidence

```sh
nix develop .#clang-mut
cmake --preset unixlike-clang-mut -Dvkexec_MULL_WORKERS=1 -Dvkexec_MULL_MINIMUM_TIMEOUT_MS=1000
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_barrier_coalescing
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_barrier_coalescing_uncovered
```

Toolchain: Clang/LLVM 22.1.8, Mull 0.35.0, Catch2 3.12.0; one worker, 1000 ms
minimum mutant timeout, 10000 ms baseline/warmup timeout. Private objects are in
`mutation-builds/barrier_coalescing/`; report prefixes are
`vkexec_core_barrier_coalescing` and `vkexec_core_barrier_coalescing_uncovered`.

Initial reports were preserved before adding any new assertion files under
`out/pr8-evidence/initial/barrier_coalescing/`. Three final pairs are preserved in
`out/pr8-evidence/final/barrier_coalescing-{1,2,3}/`, with raw SQLite, text/JSON,
profiles, hashes and metadata. The [committed JSON](mutation-barrier-coalescing-baseline.json)
retains both complete populations, per-function expressions/classifications,
source/test/config/executable/toolchain hashes, versions, exact test selection,
revision/dirty state and repeated-run comparisons. Initial tool hashes were
recovered from the same immutable Nix store paths; their versions were captured
at initial execution. Committed paths normalize the checkout root only.

The earlier scopes' exact ID/status regressions, regular debug suite, formatting
and infrastructure checks are recorded in the JSON verification section.
