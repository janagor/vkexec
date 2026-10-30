# Helper switch gap-region attribution reproduction

Standalone C++ fixture: no Vulkan, Catch2, vkexec library, or parent build.
Use the repository's pinned Clang/LLVM 22.1.8 and Mull 0.35.0 shell, from the
repository root:

```sh
nix develop .#clang-mut
python3 test/mutation/helper_attribution/run.py
```

`run.py` builds six executables at `-O0` with Mull's frontend,
`-fprofile-instr-generate -fcoverage-mapping -g -grecord-command-line`.
It reuses `scripts/compare_mull_coverage.py` for automatic, explicit raw, and
explicit indexed profile selection. Output defaults to `out/pr10-evidence/fixture`;
`--build` and `--artifacts` select separate directories. Commands, executable
SHA-256, profiles, profile counts, SQLite reports, full debug logs, LLVM exports,
and individual outcomes are retained. `matrix.json` collects the comparisons.
The PR 3 fixture and its `--verify-fixture` contract remain independent.

`argc` supplies a runtime offset to exercise all nine usage/access combinations.
Both true and false equality results are checked, so either changed comparison
would fail the executable. `negative_control`, a volatile function pointer,
retains an emitted but uncalled equality. The script asserts the exact normalized
indexed mutant IDs and statuses for every variant: two killed and three uncovered for each switch variant, six
killed and one uncovered for the if control. This includes both anomalous
`valid` comparisons and the unexecuted control, and intentionally fails if a
future Mull update changes those outcomes. Source-span changes require an
explicit review of the expectations in `run.py`. Two oracle equalities in
`exercise` are also mutated and killed, demonstrating that coverage filtering
remains active.

Measured populations (discovered / killed / survived / uncovered):

| Variant | Automatic | Raw | Indexed |
| --- | --- | --- | --- |
| Single TU, constexpr | 5 / 2 / 0 / 3 | 5 / 2 / 0 / 3 | 5 / 2 / 0 / 3 |
| Multi TU, constexpr | 5 / 0 / 0 / 5 | 5 / 0 / 0 / 5 | 5 / 2 / 0 / 3 |
| Single TU, ordinary inline | 5 / 2 / 0 / 3 | 5 / 2 / 0 / 3 | 5 / 2 / 0 / 3 |
| Multi TU, ordinary inline | 5 / 0 / 0 / 5 | 5 / 0 / 0 / 5 | 5 / 2 / 0 / 3 |
| Out of line | 5 / 2 / 0 / 3 | 5 / 2 / 0 / 3 | 5 / 2 / 0 / 3 |
| Single TU, constexpr, if control | 7 / 6 / 0 / 1 | 7 / 6 / 0 / 1 | 7 / 6 / 0 / 1 |

In every switch variant, the two executed `valid` equalities are incorrectly
uncovered; the third uncovered mutant is the intentional negative control.
The if control changes only the fixture's control-flow syntax and adds two
usage equalities to the population. All six executed mutants are killed.
It is a diagnostic control, not a proposed production rewrite.

Inspect `single/indexed.log`: Mull prints a zero-count range spanning the switch
body, with bit 31 still present in the end column. In
`single/coverage-export.json`, the same span is a gap region (`kind == 3`),
not an unexecuted code region. It encloses narrower, executed code regions.
The [investigation](../../../docs/development/mutation-helper-attribution.md)
contains real-binary evidence and an upstream issue description.
