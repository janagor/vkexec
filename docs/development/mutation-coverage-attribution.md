# Mull coverage attribution

Measured 2026-10-29–30 on x86_64-linux, Clang/LLVM **22.1.8**, Mull **0.35.0**.
Starting revision `e058c7b8cff193617ba413afbf4d6ad140c5b2d4`, with this PR's
working-tree diagnostic and CMake changes; not a finalized committed baseline.
The range implementation, tests, `mull.yml`, and pinned toolchain are unchanged.

The verified workaround is to collect a fresh LLVM raw profile from the existing
`[ranges]` executable, merge it with matching `llvm-profdata`, and pass the indexed
profile through `--coverage-info`. Keep mutation and coverage instrumentation on
both `vkexec` and `vkexec_core_tests`, including `MULL_CONFIG` during compilation
and cache bypass. No production implementation or range assertions changed.

## Same-executable comparison, performed first

The original configure/build command reproduced **47 discovered, 4 killed,
43 uncovered**. Before changing CMake or creating the standalone fixture, the
same executable and `[ranges]` filter produced an explicit indexed-profile run
of **38 killed, 2 survived, 7 timed out, 0 uncovered**. All 47 raw IDs matched the
original SQLite report. All 43 previously uncovered mutants were evaluated.

A subsequent controlled automatic/raw/indexed comparison on the unchanged
executable retained full debug logs and a SHA-256 fingerprint:

| Profile supplied to Mull | Discovered | Killed | Survived | Timed out | Uncovered |
| --- | ---: | ---: | ---: | ---: | ---: |
| Automatic warmup raw profile | 47 | 4 | 0 | 0 | 43 |
| Explicit manually collected raw profile | 47 | 4 | 0 | 0 | 43 |
| Indexed merge of that same manual raw file | 47 | 37 | 2 | 8 | 0 |

The raw and indexed tests use the **same collected profile**, eliminating test
selection, collection environment, and warmup behavior as explanations. The
profile representation changes attribution. This is not evidence that the code
was unexecuted, nor a reason to add resource-range tests here.

Reproduce the comparison from the clang-mut shell:

```sh
cmake --preset unixlike-clang-mut
cmake --build out/build/unixlike-clang-mut --target vkexec_core_tests
python3 scripts/compare_mull_coverage.py \
  out/build/unixlike-clang-mut/test/vkexec_core_tests \
  out/build/unixlike-clang-mut/mull.yml \
  out/build/unixlike-clang-mut/mutation-reports/coverage-attribution/comparison '[ranges]'
```

The helper compares all raw IDs within the same executable, checks subprocess
exits, removes its old named profiles/reports, and preserves commands, full debug
logs, both profiles, counter dumps, SQLite, and JSON. `MULL_LIBRARY_PATH` supplies
the existing dependency scanner paths. If absent, export those directories from
`vkexec_MULL_LIBRARY_PATHS` joined with colons before invoking the helper.
`--debug-coverage` alone does not enable debug-level region messages: also use
`--debug`. Missing coverage mappings in Vulkan/libc/libstdc++ are expected;
resource-range mappings reside in the test executable.

## Isolated instrumentation matrix

Only application of `-fpass-plugin` changes. Both compiled targets retain `-g`,
`-O0`, `-grecord-command-line`, `-fprofile-instr-generate`, `-fcoverage-mapping`,
config propagation and cache bypass. Separate directories were built from scratch.

| Variant | Mutated targets | Discovered | Killed | Survived | Uncovered |
| --- | --- | ---: | ---: | ---: | ---: |
| A | library and tests | 47 | 4 | 0 | 43 |
| B | tests only | 47 | 4 | 0 | 43 |
| C | library only | 0 | 0 | 0 | 0 |

A and B have identical raw mutant IDs. C's runner says `No mutants found` and
creates no report; the existing audit validator correctly fails rather than
accepting an empty audit. This is a diagnostic outcome, not a successful audit.
An explicit indexed comparison in B also evaluates all 47 (39 killed, 2 survived,
6 timed out). Reducing instrumentation does not fix the discrepancy.

The matrix is retained as historical experiment evidence. The temporary
`vkexec_MULL_MUTATION_TARGETS` cache option was removed after the experiments;
normal mutation builds always instrument both targets. To repeat B or C, use an
isolated diagnostic checkout and build directory, applying the Mull pass only to
the selected target while retaining coverage flags on both targets. The measured
builds reused local FetchContent *sources* through
`FETCHCONTENT_SOURCE_DIR_<NAME>`, never compiled objects or libraries.

`llvm-nm -C` inspected all relevant library objects, the archive and test objects.
No affected range definitions occur in the compiled library or its archive.
`normalize`, `overlaps`, and `split` are emitted as weak definitions by eight test
TUs: scheduler_graph, scheduler_pass, resource_ranges, image_region,
pass_detail_resource, buffer_region, pass_detail, and scheduler tests. The examined
`contains<unsigned long>` definition occurs only in resource_ranges tests.
`llvm-readobj --section-groups` confirms COMDAT function sections. The linked
executable retains one weak wrapper for each affected overload/instantiation.
`llvm-objdump -drC` shows that the linked `normalize(buffer_use)` wrapper selects
Mull's original and mutant implementations via `getenv` and indirect calls.
Indexed-profile kills demonstrate execution of those selected mutant bodies.
There is no evidence that removing the library pass solves a linker mismatch.

Reinspect with:

```sh
llvm-nm -C out/build/mull-coverage-control/src/vkexec/libvkexec.a
llvm-nm -C out/build/mull-coverage-control/test/CMakeFiles/vkexec_core_tests.dir/core/resource_ranges_tests.cpp.o
llvm-nm -C out/build/mull-coverage-control/test/vkexec_core_tests
llvm-readobj --section-groups \
  out/build/mull-coverage-control/test/CMakeFiles/vkexec_core_tests.dir/core/resource_ranges_tests.cpp.o
llvm-objdump -drC out/build/mull-coverage-control/test/vkexec_core_tests
```

## Standalone reproduction and cause

The [fixture instructions](../../test/mutation/coverage_attribution/README.md)
cover single-TU inline code, multi-TU inline/templates, and out-of-line overlaps
with header templates as a control. All were executed with the same flags,
operators and inputs, without vkexec/Vulkan/Catch2. Single TU evaluates 8/10
mutants under either format; multi-TU evaluates 0/10 with raw and 8/10 with indexed;
out-of-line evaluates 4/10 with raw and 8/10 with indexed. The two mutants in an
intentionally uncalled function remain uncovered with indexed coverage. Thus
merging does not simply disable the coverage filter.

LLVM's own counter dump of the multi-TU raw profile includes duplicate name/hash
records for inline/template instantiations. Indexed merging combines those
records. Mull's raw coverage debug output nevertheless marks the executed header
function bodies as zero-hit ranges; its indexed output retains only the uncalled
function as uncovered. The source-region coverage investigation independently establishes
execution in the real project.

The supported root-cause boundary is **Mull 0.35's raw-profile parsing/mapping for
multi-TU inline/template coverage**. The exact erroneous operation in its Rust
profile dependency has not been isolated or patched. Do not claim a proven
LLVM compiler defect or a general fix to all Mull mapping limitations.
Mull's [tagged runner source](https://github.com/mull-project/mull/blob/992abfa6fdccb2d3bf4186a254376147903668ba/rust/mull-tools/mull-runner.rs)
passes the automatic raw file directly to its filters;
[coverage_filter.rs](https://github.com/mull-project/mull/blob/992abfa6fdccb2d3bf4186a254376147903668ba/rust/mull-filters/src/coverage_filter.rs)
uses `llvm_profparser::parse` and treats zero-hit regions as uncovered.
The [upstream issue draft](mutation-coverage-attribution-upstream.md) is prepared
for review; no issue was posted.

## Corrected baseline and limits

Both existing targets now collect and merge a clean per-audit profile before
running Mull with `--coverage-info`. Configuration requires matching llvm-profdata.
Normal builds are unaffected. Run targets sequentially to reduce timing noise:

```sh
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_ranges
cmake --build out/build/unixlike-clang-mut --target vkexec_mutation_ranges_uncovered
ctest --test-dir out/build/unixlike-clang-mut \
  -R 'vkexec.mull_(report_validator|coverage_diagnostics)' --output-on-failure
```

The final sequential measurements were:

| Measurement | Historical uncovered audit | Corrected ranges | Corrected ranges_uncovered |
| --- | ---: | ---: | ---: |
| Discovered/reported | 47 | 47 | 47 |
| Killed | 4 | 40 | 37 |
| Survived | 0 | 2 | 2 |
| Timed out | 0 | 5 | 8 |
| Uncovered | 43 | 0 | 0 |
| Evaluated/discovered | 8.51% | 100% | 100% |
| Score, killed/(killed+survived) | 100% of 4 | 95.24% of 42 | 94.87% of 39 |
| Mull runner time | historical 234 ms | 796 ms | 718 ms |

No crashes or abnormal exits occurred. Exact per-mutant outcomes, normalized IDs,
executable hashes and all experiment counts are in the
[measurement artifact](mutation-coverage-attribution.json). Relevant commands,
raw/indexed region debug output, symbols and final target logs are retained in
[evidence excerpts](mutation-coverage-attribution-evidence.txt). Reproduction commands write full logs
and SQLite/profiles under `out/build/unixlike-clang-mut/mutation-reports/coverage-attribution`,
`out/build/mull-coverage-{control,tests-only,library-only}`, and `out/build/mull-coverage-fixture/reports`; generated binaries and
large profiles are not committed.

All IDs in both corrected reports equal the original 47 IDs. All 43 formerly
uncovered mutants received execution outcomes. In the corrected `ranges` report,
`normalize(buffer_use)` has seven killed and one timeout, interval `overlaps`
has four killed, and `split` has four killed. `normalize_subresources` has seven
killed and one survivor. The two survivors are `cxx_gt_to_ge` at 75:28 and 114:39.
Their behavioral/equivalence analysis belongs to the next test-improvement PR.

**Timeout limitation:** Mull 0.35 uses `--timeout` for warmup and baseline; mutant
timeout is `max(10 * baseline_duration, --minimum-timeout)`, with a default minimum
of 30 ms. Thus the existing `--timeout 10000` is not a fixed ten-second mutant
budget. Even sequential runs showed killed/timeout variation; initial indexed
runs yielded 38/2/7 and 37/2/8. This PR leaves timeout policy unchanged to isolate
the attribution fix. Timeouts count as evaluated but never as kills or in the
mutation-score denominator. The corrected attribution is reproducible; these
scores are timing-sensitive observations, not regression thresholds.

Both final original `[ranges]` runs passed **72,918 assertions in 53 cases**.
Both Python CTest diagnostic suites passed (report-validator and coverage
attribution diagnostics), also verified directly. The fixture's two uncalled
mutants remained uncovered in all indexed runs. No new behavioral range tests,
DAG/synchronization mutation scope, GPU execution, production algorithm changes,
or CI thresholds were introduced. The old 4/47 audit remains historical and
experimental; use the indexed-profile results for subsequent work, retaining
the timeout caveat.

## Revision metadata

`source_revision` and `source_dirty` in the measurement artifact describe the
historical measured working tree. They must not be rewritten to imply a clean
measurement at a later revision. `measurement_source_revision` explicitly names
that historical base. `finalized_source_commit` records implementation revision
`b4863691213738ddd251ea8a01c97b05678e605e`, which includes the indexed-profile fix
and standalone regression check. `finalization_status` is `implementation_committed`.
This documentation commit records that existing implementation revision rather than
a self-referential hash; it does not relabel the historical working-tree runs as
clean measurements at the finalized revision.

The comparison helper now records the effective `LLVM_PROFILE_FILE` and
`MULL_CONFIG` for every command, including explicit unsets when absent. Mull's
automatic child profile path is supplied internally by the runner and appears in
`automatic.log` as `coverageFile`; it is not a parent environment override.
Historical command excerpts predate this logging enhancement. The opt-in
`--verify-fixture` check requires eight called mutations to be killed/survived and
two mutations in the uncalled fixture function to remain uncovered with indexed
coverage. It imposes no expected outcomes on raw profiles.

Review cleanup validation: after removing the diagnostic instrumentation option,
both targets again evaluated 47/47 with zero uncovered. `ranges` reported
39 killed, 2 survived, 6 timed out; `ranges_uncovered` reported
37 killed, 2 survived, 8 timed out. Both original suites passed
72,918 assertions in 53 cases, and both Python CTest diagnostic suites passed.
All three standalone variants passed `--verify-fixture`. Negative checks against
copies of the indexed report correctly rejected a called mutant marked uncovered
and an uncalled mutant marked killed. These review measurements are recorded
separately from the historical experiment results in the JSON artifact.
