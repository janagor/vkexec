# Draft upstream issue: raw versus indexed multi-TU coverage in Mull 0.35

Prepared for review; not submitted.

**Title:** Mull 0.35 marks executed multi-TU inline/template mutations uncovered
with raw profiles; indexing the same profile restores attribution

Environment: Linux x86_64, Clang and LLVM tools 22.1.8, Mull 0.35.0 official Debian
trixie LLVM 22.1.8 package repackaged by Nix. Mull source inspected at tag 0.35.0,
commit `992abfa6fdccb2d3bf4186a254376147903668ba`.

Attach `test/mutation/coverage_attribution/` and the repository's
`scripts/compare_mull_coverage.py`. The fixture is an independent CMake project
with no Vulkan or test-framework dependency. See its README for full commands;
compile with:

```sh
MULL_CONFIG=/absolute/build/mull.yml clang++ -g -O0 -grecord-command-line \
  -fpass-plugin="$MULL_PLUGIN" -fprofile-instr-generate -fcoverage-mapping \
  -c a.cpp -o a.o
```

All other fixture TUs use the same options. Link with
`-fprofile-instr-generate`. CMake supplies the necessary single/multi/out-of-line
preprocessor definitions. The YAML enables `cxx_comparison` and `cxx_boundary`
and includes only `ranges.hpp` and `out_of_line.cpp`.

The helper executes each binary once with `LLVM_PROFILE_FILE=manual.profraw`,
merges that exact file using LLVM's `llvm-profdata merge -sparse`, then compares:

```sh
MULL_CONFIG=/absolute/build/mull.yml mull-runner-22 \
  --allow-surviving --include-not-covered --timeout 10000 \
  --debug --debug-coverage --reporters SQLite \
  --report-dir /absolute/reports --report-name raw \
  --coverage-info /absolute/reports/manual.profraw /absolute/build/multi
# Repeat as report-name indexed with manual.profdata; omit --coverage-info for automatic.
```

Add `--ld-search-path` per runtime directory where needed (the helper reads
`MULL_LIBRARY_PATH`). Use fresh report names or remove prior SQLite outputs;
Mull appends otherwise. Preserve pipeline exit status if teeing output.

Expected: raw and indexed representations of the same execution should select
the same eight exercised mutations and leave the two uncalled mutations uncovered.

Actual:

| Experiment | Automatic/raw | Indexed |
| --- | --- | --- |
| Single TU inline + two template types | 6 killed, 2 survived, 2 uncovered | same |
| Multiple TUs, inline + same template types | 10 uncovered | 6 killed, 2 survived, 2 uncovered |
| Out-of-line non-template + header templates | 3 killed, 1 survived, 6 uncovered | 6 killed, 2 survived, 2 uncovered |

Discovered raw IDs are identical across formats for each binary. No timeouts.
The out-of-line non-template mutations evaluate with raw profiles, whereas the
header template mutations recover with indexed profiles. No `-fno-inline`
workaround or production implementation relocation is needed.

Attach `reports/{single,multi,out_of_line}/{automatic,raw,indexed}.log`,
`manual.profraw`, `manual.profdata`, the corresponding executables, and
`comparison.json`/`commands.txt` from the fixture helper. Relevant normalized
source-region diagnostics and commands are also preserved in
[local evidence](mutation-coverage-attribution-evidence.txt).

For example the multi-TU raw diagnostics mark the executed `overlaps` body at
`ranges.hpp:9:54:9:84` and its comparisons as uncovered. The indexed diagnostics
retain only `ranges.hpp:4:58:4:75` (the intentionally uncalled function).
LLVM's raw counter dump shows duplicate inline/template name/hash records with
nonzero counts; merging produces combined indexed records.

The observed boundary is raw-profile decoding/mapping through
`llvm_profparser` 0.15 in `coverage_filter.rs`, not absent execution. The exact
dependency defect is not yet established. In the real vkexec case, identical
mutant IDs change from 4 evaluated / 43 uncovered to 47 evaluated / 0 uncovered
when indexed profiles are supplied. Workaround: use matching LLVM profdata to
index the raw profile before passing `--coverage-info`.
