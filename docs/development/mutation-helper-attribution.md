# Helper mutation coverage attribution (PR 10)

Mull 0.35.0 incorrectly classifies four executed equality comparisons as
`not_covered` because its coverage path treats LLVM switch **gap regions** as
zero-count code ranges. This is reproducible in a standalone single-TU ordinary
function, as well as constexpr, multi-TU, and out-of-line variants. Indexed
profiles fix the separate general multi-TU problem, but do not fix this one.
No production code, audit instrumentation, mutation operators, test selection,
validator policy, or baseline classifications are changed by this investigation.

Measured on 2026-10-30, Linux x86_64, Clang/LLVM 22.1.8 and Mull 0.35.0,
from clean revision `09e4b5a6fc5b91c223439c29f8062bda1da0ae3a` before adding
these diagnostics. The initial PR 9 runner comparison passed without fingerprint
mismatches: **56 discovered, 51 killed, 1 survived, 4 uncovered**. Evaluation
coverage remains 52/56; mutation score remains 51/52. The four are not kills.

The [machine-readable evidence](mutation-helper-attribution.json) retains the
normalized IDs, actual region tuples, function counts, enclosing gaps, Mull
exclusion ranges, baseline fingerprints, executable hashes, and fixture results.
Raw local evidence is under `out/pr10-evidence/`; it is generated, not committed.
The original PR 8 historical JSON and PR 9 baseline remain intact.

## Real executable and profile association

The private binary is
`out/build/unixlike-clang-mut/mutation-builds/resource_helpers/test/vkexec_core_tests`.
Its SHA-256 during diagnostics was
`1e7704b58ec24308c5a7d419d4ec85ee1e14fa8b8be55acfd1813e1c0158fc2c`.
Selection was exactly `[resource-helpers] --order lex --abort`.
The initial runner removed prior scope outputs before generating each report and
profile. `diagnostics/baseline.profdata` and `.profraw` are copies of that audit's
inclusive profiles; `coverage-export.json` and `profile-counts.txt` use the same
executable and copied indexed profile. The separate diagnostic comparison
regenerates raw/indexed profiles from that executable and verifies its SHA-256
before and after all three Mull modes. It reproduces the same four indexed
uncovered IDs. Thus stale profiles or a different executable are unnecessary to
explain the anomaly.

All four IDs have prefix
`cxx_eq_to_ne:include/vkexec/detail/resource_state_tracker_helpers.hpp:`:

| ID suffix | Function | Innermost executed code region | Count | Enclosing gap |
| --- | --- | --- | ---: | --- |
| `83:23:83:25` | valid(image_use) | 82:3–83:47 | 144 | 78:22–90:16 |
| `85:23:85:25` | valid(image_use) | 84:3–85:48 | 27 | 78:22–90:16 |
| `105:23:105:25` | valid(buffer_use) | 104:3–105:47 | 189 | 97:22–110:16 |
| `107:23:107:25` | valid(buffer_use) | 106:3–107:48 | 27 | 97:22–110:16 |

LLVM export reports one function record for each overload, named
`_ZN6vkexec6detail5validERKNS_9image_useE` and
`_ZN6vkexec6detail5validERKNS_10buffer_useE`. `llvm-profdata show --all-functions
--counts` reports respectively:

- Image: hash `0x0d9d653cf79465da`, 11 counters, function count 288,
  block counts `[0, 63, 27, 27, 27, 27, 36, 27, 27, 27]`.
- Buffer: hash `0x0e4bb3ea8b2babe4`, 12 counters, function count 297,
  block counts `[0, 27, 27, 27, 27, 27, 27, 27, 54, 27]`.

`symbols.txt` retains the linked weak symbols and profile-counter symbols.
`valid-disassembly.txt` shows the linked dispatchers calling their corresponding
`mull__..._original` functions and selecting exactly the two mutant-ID symbols
for each overload through environment guards. This connects the emitted mutants
to the named functions; the diagnosis does not depend on inferring identity
from a total line count. LLVM does not expose per-TU identity for all linked
COMDAT copies here, and that is not needed to explain the observed exclusions.

## Mull's exclusion decision

With both `--debug` and `--debug-coverage`, indexed Mull prints these ranges
among its zero-count exclusions (checkout prefix omitted):

```text
mull-coverage: include/vkexec/detail/resource_state_tracker_helpers.hpp:78:22:90:2147483664
mull-coverage: include/vkexec/detail/resource_state_tracker_helpers.hpp:97:22:110:2147483664
```

Both contain the corresponding mutation points. LLVM exports those same spans
as `[78,22,90,16,0,0,0,3]` and `[97,22,110,16,0,0,0,3]`: **kind 3, gap**,
with zero counts, enclosing positively executed code regions. Mull's printed
end column is `16 + 2^31`, consistent with an unmasked gap marker. This is not a
missing function identity or a zero count on the innermost executed expression.
The indexed report has 52 evaluated mutants, excluding a global coverage-load
failure. Automatic and explicit raw profiles instead classify all 56 uncovered,
the separate already-known multi-TU raw-profile limitation.

The local upstream source checkout at tag `0.35.0`, revision
`992abfa6fdccb2d3bf4186a254376147903668ba`, explains the final decision:
`rust/mull-filters/src/coverage_filter.rs` loads a report from `llvm_profparser`,
puts every zero-hit location in `uncovered_ranges`, and `covered()` rejects a
point contained in **any** such range. It does not prefer an overlapping
positively executed code region or distinguish gaps there. The debug output
establishes that the gap spans have reached that filter as zero-hit locations.
This investigation isolates the externally observable parser/filter limitation;
it does not claim which internal dependency change is the best upstream fix.

## Standalone reproduction and controls

See [fixture instructions](../../test/mutation/helper_attribution/README.md)
for the complete matrix and a one-command runner. The fixture uses runtime
`argc` input and checks both valid and invalid access combinations.
Every indexed switch variant has exactly five mutants: two killed oracle
comparisons, two incorrectly uncovered switch comparisons, and one genuinely
uncalled comparison that stays uncovered. Single-TU constexpr, single-TU
ordinary inline, multi-TU constexpr, multi-TU ordinary inline, and out-of-line
implementations all reproduce. Removing constexpr, duplicate header emission,
and multi-TU linking therefore does not remove this failure.

An additional single-TU `if` control preserves the semantic checks and removes
the switch: all six executed comparisons are killed in all profile modes, while
the uncalled comparison remains uncovered. Its seven-mutant population includes
two additional usage comparisons; it is not compared as an identical population.
This control is fixture-only. Rewriting vkexec's switch to improve mutation
scores would conceal a tool limitation and is not proposed. No optimization
change is necessary: the smallest reproduction already fails at `-O0`.

## Reproduce the real-binary diagnostics

From the repository root, in `nix develop .#clang-mut`, use fresh audit and
comparison destinations:

```sh
python3 scripts/ci/run_mutation_audits.py resource_helpers \
  --policy regression --artifacts out/pr10-evidence/baseline
mkdir -p out/pr10-evidence/diagnostics
binary=out/build/unixlike-clang-mut/mutation-builds/resource_helpers/test/vkexec_core_tests
profile=out/pr10-evidence/baseline/resource_helpers/inclusive.profdata
sha256sum "$binary"
llvm-cov export "$binary" -instr-profile="$profile" \
  > out/pr10-evidence/diagnostics/coverage-export.json
llvm-profdata show --all-functions --counts "$profile" \
  > out/pr10-evidence/diagnostics/profile-counts.txt
python3 scripts/compare_mull_coverage.py "$binary" \
  out/build/unixlike-clang-mut/mutation-builds/resource_helpers/mutation-configs/resource_helpers.yml \
  out/pr10-evidence/diagnostics/modes -- '[resource-helpers]' --order lex --abort
python3 scripts/diagnose_mull_coverage.py \
  out/pr10-evidence/baseline/resource_helpers/inclusive.sqlite \
  include/vkexec/detail/resource_state_tracker_helpers.hpp \
  include/vkexec/detail/resource_state_tracker_validation.hpp \
  out/pr10-evidence/diagnostics/coverage-export.json \
  --mull-log out/pr10-evidence/diagnostics/modes/indexed.log \
  --json-output out/pr10-evidence/diagnostics/source-comparison.json
llvm-nm --defined-only "$binary" > out/pr10-evidence/diagnostics/symbols.txt
llvm-objdump --line-numbers \
  --disassemble-symbols=_ZN6vkexec6detail5validERKNS_9image_useE,_ZN6vkexec6detail5validERKNS_10buffer_useE \
  "$binary" > out/pr10-evidence/diagnostics/valid-disassembly.txt
python3 test/mutation/helper_attribution/run.py
```

The diagnostic CLI now accepts the complete multi-source allowlist and can
correlate Mull debug ranges. It retains the original classification semantics,
adding full selected region tuples, enclosing gap regions and function counts.
It does not convert source coverage into a mutant execution outcome.

## Upstream issue draft

**Title:** Mull 0.35.0 filters executed switch comparisons as uncovered using
LLVM gap regions (Clang 22.1.8, indexed profile).

**Expected:** Evaluate the two runtime-executed equality mutants in `valid` and
keep the emitted, uncalled negative control uncovered.

**Actual:** Even with a freshly merged indexed profile, both equalities remain
uncovered. LLVM shows nonzero innermost code regions; Mull logs an enclosing
zero-count gap as an uncovered range, including the gap bit in its end column.
A single-TU ordinary function suffices; no Catch2, Vulkan, templates, or vkexec
library are required. The fixture's if control evaluates every executed mutant
and preserves the uncovered negative control.

**Attachments:** `test/mutation/helper_attribution/`, `fixture/matrix.json`,
`fixture/single/{commands.txt,indexed.log,indexed.sqlite,manual.profraw,
manual.profdata,coverage-export.json}`, compiler/Mull versions and binary SHA.
The committed evidence JSON supplies portable normalized outcomes; regenerate
the binary and raw reports with `run.py` for the recipient's checkout paths.
No upstream issue has been submitted by this PR.

## Validation

Three complete covered/inclusive helper audits (`baseline`, `validation`,
`repeat-3`) produced identical normalized IDs and statuses in both modes:
52 covered-mode mutants and 56 inclusive-mode mutants, with 51 killed,
1 survived, and 4 uncovered in the inclusive reports. The evidence JSON records
matching outcome hashes for all three pairs.

All five scope comparisons passed PR 9 regression policy with unchanged
populations, zero regressions, no fingerprint mismatches, and no timeouts,
crashes, abnormal exits, or allowlist violations:

| Scope | Killed | Survived | Uncovered |
| --- | ---: | ---: | ---: |
| ranges | 47 | 0 | 0 |
| dag | 36 | 2 | 19 |
| resource_state | 54 | 2 | 0 |
| resource_helpers | 51 | 1 | 4 |
| barrier_coalescing | 15 | 1 | 0 |

The comparator reports `tests_changed: true` after adding the standalone fixture,
as intended by its existing test-change policy; instrumentation and execution
fingerprints remain comparable. No fingerprint policy was weakened and no
baseline was rewritten.

The three mutation infrastructure CTests passed, including the extended gap
correlation diagnostic tests and audit registration. All 23 PR 9 comparator and
runner tests passed. The normal debug core executable passed 221 cases and
94,393 assertions, with one queue-capability skip; its contract-death tests emit
expected child-process abort diagnostics. All six standalone variants completed
all three coverage modes and passed their negative/positive control checks.
`clang-format --dry-run --Werror` and `git diff --check` passed.

No instrumentation fix or rebaseline is warranted: keep these four limitations
visible and separate from PR 11 operator expansion.
