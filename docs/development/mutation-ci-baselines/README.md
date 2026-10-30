# CI mutation baselines

These snapshots are regenerated measurements of clean PR 8 commit
`3504e0ea54c04d333d72dcc6fd92f29413573594`, not relabeled dirty historical
measurements. Each includes its actual revision, separate execution/source/test
fingerprints and the complete inclusive mutant population. IDs retain source
paths, mutators and coordinates, with only the measured checkout prefix removed.

| Scope | Total | Killed | Survived | Uncovered |
|---|---:|---:|---:|---:|
| ranges | 47 | 47 | 0 | 0 |
| dag | 57 | 36 | 2 | 19 |
| resource_state | 56 | 54 | 2 | 0 |
| resource_helpers | 56 | 51 | 1 | 4 |
| barrier_coalescing | 16 | 15 | 1 | 0 |

Reproduction uses Clang/LLVM 22.1.8 and Mull 0.35.0 from the committed Nix lock,
`unixlike-clang-mut`, the ten public audit targets, one worker, a 10000 ms warmup
timeout and a common 1000 ms minimum mutant timeout. Ranges retains the same
47 killed classifications as the historical final 250 ms runs. No timeout,
crash or abnormal-exit outcomes are accepted as kills.

The baseline writer validates each covered report against its inclusive partner
and checks manifest report hashes and fingerprints against the clean checkout.
Exact normalized IDs and statuses are also checked against each historical final
matrix; historical evidence files remain unchanged. Local evidence is retained
under `out/pr9-evidence/`; it is not a hosted GitHub Actions execution. Hosted
reproducibility must be established before enabling regression gating.

See [baseline review](../mutation-audit-scopes.md#ci-baseline-review) for explicit
updates and [CI execution](../mutation-testing.md#automated-ci-audits) for policies,
artifacts and reproduction commands.

Local PR 9 validation completed with 23 standard-library unit/regression tests,
all three existing mutation-infrastructure CTests, all ten audit reports, and a
full run of the new runner in regression mode. Every scope is comparable and
passes with zero regressions. Evidence: `clean-baseline/manifest.json`,
`regression-run/manifest.json`, `regression-run/summary.md`, and
`historical-comparison.json` under the local evidence directory. Failure retention
is exercised by injecting a covered-target failure and checking that its
inclusive target and the remaining scopes still execute with artifacts retained.
A CLI fixture independently confirms killed-to-survived returns exit 2 under
regression policy and remains reported under report policy.
