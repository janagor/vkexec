# Mull 0.35 raw-profile attribution reproduction

Standalone: no vkexec, Vulkan, Catch2, or parent CMake build is needed.
Use Clang 22.1.8, Mull 0.35.0 and LLVM 22.1.8 from the repository's
`nix develop .#clang-mut` shell. From the repository root:

```sh
cmake -S test/mutation/coverage_attribution -B out/build/mull-coverage-fixture \
  -G Ninja -DCMAKE_CXX_COMPILER=clang++
cmake --build out/build/mull-coverage-fixture
for target in single multi out_of_line; do
  python3 scripts/compare_mull_coverage.py \
    "out/build/mull-coverage-fixture/$target" out/build/mull-coverage-fixture/mull.yml \
    "out/build/mull-coverage-fixture/reports/$target" --verify-fixture
done
```

The helper needs only Python's standard library, `llvm-profdata`, and Mull. It
uses `MULL_RUNNER` (or `mull-runner-22`) and `MULL_LIBRARY_PATH` from the shell.
It checks command exits, compares raw IDs within each executable, replaces old
reports (Mull otherwise appends), and retains exact commands, profiles, counts,
SQLite reports, and full debug logs. Use a separate output directory per binary.
`--verify-fixture` is an opt-in regression check, outside the default CTest suite.
It requires exactly eight distinct called mutants to be killed or survived and
two mutants at the uncalled function to stay uncovered in the indexed report.
It does not assert raw-profile failure, so future Mull fixes are accepted.
`commands.txt` includes effective `LLVM_PROFILE_FILE` and `MULL_CONFIG` assignments
or explicit unsets for each command.
Both `--debug` and `--debug-coverage` are needed to print uncovered region details.

All three executables exercise the same overlapping, adjacent, and disjoint
inputs, using ordinary return codes rather than a test framework. They mutate
only comparison/boundary operators in `ranges.hpp` and `out_of_line.cpp`.
`single` includes inline and template implementations in one TU. `multi` calls
them from `a.cpp` and `b.cpp`, instantiating templates for `int` and `long`.
`out_of_line` moves only the non-template overlap implementation out of the
header; templates deliberately remain in the header as an internal control.
The uncalled, emitted function must retain two uncovered mutants in every
indexed run; it detects a workaround that merely disables coverage filtering.

Measured counts (discovered / killed / survived / uncovered):

| Binary | Automatic | Explicit raw | Explicit indexed |
| --- | --- | --- | --- |
| single | 10 / 6 / 2 / 2 | 10 / 6 / 2 / 2 | 10 / 6 / 2 / 2 |
| multi | 10 / 0 / 0 / 10 | 10 / 0 / 0 / 10 | 10 / 6 / 2 / 2 |
| out_of_line | 10 / 3 / 1 / 6 | 10 / 3 / 1 / 6 | 10 / 6 / 2 / 2 |

No fixture mutant timed out. Out-of-line overlaps works with raw coverage;
header template mutations recover only with indexed coverage. The survivors are
intentional outcomes of this small fixed input set, not a claim of equivalent
mutants. Do not add behavioral assertions to make the score look better.

For a report-format-only comparison, supply the same `manual.profraw` directly
with `--coverage-info`, then merge that file using
`llvm-profdata merge -sparse manual.profraw -o manual.profdata` and supply the
indexed file. See `commands.txt` emitted by the helper for complete invocations.
The [investigation](../../../docs/development/mutation-coverage-attribution.md)
records the real-project controls and verified workaround.
