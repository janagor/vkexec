# Controlled OR-operator expansion (PR 11)

This is an isolated experiment on `resource_helpers`, not a sixth registered
scope or a replacement for a PR 9 baseline. Only `cxx_or_to_and` and
`cxx_or_assign_to_and_assign` are added. The combined configuration also enables
`cxx_comparison` and `cxx_boundary`. Canonical configurations, presets, scope
registration, production code, and baseline snapshots are unchanged.

## Reproduction and provenance

Enter `nix develop .#clang-mut`, then run from a clean committed source checkout:

```sh
python3 scripts/mutation_operator_experiment.py --variant or --revision HEAD \
  --artifacts out/pr11-evidence/or-initial
python3 scripts/mutation_operator_experiment.py --variant combined --revision HEAD \
  --artifacts out/pr11-evidence/combined-initial
```

Use fresh artifact destinations. The script checks Clang/LLVM 22.1.8 and Mull
0.35.0, then compiles a tiny runtime fixture. Both operators must emit exactly
one mutant and each must be killed; recognizing a configuration name alone is
insufficient. Unknown operators, extra YAML keys, and widened source filters fail
validation.

A detached temporary worktree contains an independent public preset build and
private scope build. The selected experiment configuration replaces only
`cmake/mull/resource_helpers.yml` there. The wrapper checks the tracked diff and
records the original and replacement configuration hashes. The audit manifest's
`source_clean: false` remains intact. `experiment.json` records the intentional
override and source revision; it is never a canonical clean-baseline claim.

The wrapper and configurations can also reside outside the clean source
checkout. This was used during implementation to measure the committed PR 10
source without committing experiment infrastructure first. The wrapper hash and
configuration hash identify that separately supplied infrastructure. Measurement
source revision is recorded in the snapshot, rather than inferred from the
branch's eventual tip.

The PR 9 runner executes both existing audit targets with `--measure-only`.
SQLite outputs are independently validated before comparison to any generated
JSON. Exact IDs retain the mutator, checkout-relative source path, and complete
source span. Duplicate IDs, unexpected paths/statuses, missing databases, and
covered/inclusive contradictions fail validation. The inclusive population must
include both allowlisted headers; the covered population must equal precisely
the evaluated subset of the inclusive population.

### Covered-output infrastructure limitation

The initial new-only audit found three mutants: both validation-header mutants
were killed, while the helper-header mutant was uncovered. The canonical covered
target consequently failed its existing requirement that *each* allowlisted
header have a covered mutant. Mull had successfully produced a covered SQLite
report with two kills, but no covered JSON was generated. The inclusive target
succeeded with all three mutants.

The wrapper preserves this runner failure and accepts only this specific
missing-source validator failure after independently validating both databases.
It writes separate `validated-covered.json` and `validated-inclusive.json`
outputs. It does not rewrite the failed audit manifest, fabricate the missing
canonical JSON, modify the canonical validator, or treat arbitrary build/runner
failures as successful experiments.

## Measurement and interpretation

| Population | Discovered | Killed | Survived | Uncovered |
| --- | ---: | ---: | ---: | ---: |
| Original PR 9 resource helpers | 56 | 51 | 1 | 4 |
| Initial OR-only experiment | 3 | 2 | 0 | 1 |
| Each of three final OR-only repeats | 3 | 2 | 0 | 1 |
| Combined experiment: original subset | 56 | 51 | 1 | 4 |
| Combined experiment: new subset | 3 | 2 | 0 | 1 |

The measured source revision is `50e9bfed0c6edc5cba6f7a54db2e95c901e95863`.
The new-only initial outcome was recorded before any test changes. No test
assertions have been added: there is no evaluated survivor demonstrating a
missing assertion. In particular, the existing disjoint `access_mask` assertion
and the existing source-scope tests should not be duplicated.

The emitted new sites are `access_mask()` at helpers line 71, and the two
`source_scope()` accumulations at validation lines 110 and 111. The
constant-expression depth-attachment OR at helpers line 143 does **not** appear
in the measured population. Source operator occurrences do not establish emitted
mutants or coverage; this experiment makes no fault-detection claim for that site.

At the uncovered `access_mask()` site, LLVM reports 135 executions of code
region 70:3–71:34, inside gap region 65:19–71:34. Mull excludes
65:19–71:2147483682. This is the PR 10 switch-gap attribution limitation,
not evidence of a missing mask assertion. The SQLite status remains uncovered.

The wrapper retains an LLVM coverage export and Mull automatic/raw/indexed debug
coverage comparison associated with the instrumented binary hash. An uncovered
mutant is classified as a known attribution limitation only when an executed
LLVM code region, a containing switch gap region, and a Mull exclusion range
all cover its location. The SQLite status remains `not_covered`. Other unclear
outcomes remain `unresolved`; human classifications never change measured
outcomes. Function names are linked LLVM symbol names where uniquely identifiable.

## Stability and compatibility

Three final OR audits at the same source revision, in fresh worktrees, produced
identical exact IDs and statuses. The initial population is also unchanged.
The comparator reports `wrapper_changed` separately from outcome stability;
changes to the wrapper do not make stable mutation outcomes a regression.
The retained repetitions report `stable: true` and `wrapper_changed: true`,
reflecting the disclosed diagnostic-collection changes.
For reproduction, repeat the OR audit with fresh worktrees and destinations. Compare exact new IDs and statuses, including the
combined experiment's new subset:

```sh
python3 scripts/mutation_operator_experiment.py --compare \
  out/pr11-evidence/or-final-1/experiment.json \
  out/pr11-evidence/or-final-2/experiment.json \
  out/pr11-evidence/or-final-3/experiment.json \
  out/pr11-evidence/combined-final/experiment.json
```

The combined experiment matched all **56 original IDs and statuses**, with no
missing, added, or changed original outcomes. Its three new IDs and statuses
matched every final OR-only repeat.

`compatibility` in the combined manifest accounts for all 56 original baseline
IDs individually, retaining every missing, added, or changed outcome. The old
and new populations remain separate. The canonical score remains 51/52; a larger
discovered population does not imply a quality regression.

Then run the five canonical scopes:

```sh
python3 scripts/ci/run_mutation_audits.py all --policy regression \
  --artifacts out/pr11-evidence/canonical
```

All five canonical scopes passed regression comparison with **zero regressions
and no fingerprint mismatches**. Canonical resource helpers remain 56 discovered,
51 killed, 1 survived, and 4 uncovered. The canonical measurement records
`source_clean: false` because experiment infrastructure and documentation were
uncommitted during validation; no baseline writer was invoked.

The 16 orchestrator unit tests also passed with Mull environment variables unset,
and the existing 23 baseline-comparison tests, three coverage-diagnostic tests,
and three report-validator tests passed.

The experimental snapshot is
[`mutation-experiments/resource_helpers_or.json`](mutation-experiments/resource_helpers_or.json).
The first final-run collector prefixed the LLVM JSON export with its command,
which prevented JSON parsing after the audit had completed. The original output
and failed collection manifest were preserved. Removing the preamble allowed
offline attribution analysis without changing either database or mutant outcome.
The wrapper now records JSON-producing commands in a separate sidecar, with a
regression test for this case.

Raw evidence is retained locally under `out/pr11-evidence/` and is not committed.
The lightweight Python tests require neither Mull nor Nix; scheduled CI does
not execute this experiment.

The new accumulation mutants provide fault-detection coverage, while the helper
OR is limited by upstream attribution and the constant stage OR is not emitted.
Promotion to a sixth canonical scope remains a separate change that must review
registration and fingerprint changes explicitly.
