# Resource-range mutation test quality (PR 4)

Measured 2026-10-30 on x86_64 Linux, branch `testing/mull-range-test-quality`,
starting at `4ca89667390ac4dabf602c64ad70db0797236bc8` (completed PR 3).
Clang/LLVM 22.1.8, Mull 0.35.0, Catch2 3.12.0, Nix `clang-mut` shell.
Production `resource_ranges.hpp` and Mull instrumentation/configuration are unchanged.

## Frozen controls and evidence

The [PR 3 record](mutation-coverage-attribution.json) remains unchanged. Its final
review counts are 39 killed / 2 survived / 6 timed out for `ranges`, and
37 / 2 / 8 for `ranges_uncovered`, each with 47 discovered/evaluated and zero
uncovered. Scores were 95.12% and 94.87%. Its earlier full corrected `ranges`
summary contains 40 / 2 / 5; these are separate measurements, not interchangeable.

Before rebuilding, local reports, SQLite databases, raw/indexed profiles and copied
configuration were preserved in `out/pr4/control`. The local pre-edit executable
checksum was `082bab42971629b262e8514aee276f2c0cde7db48b10bb7c6c1504f46eb3428e`,
matching PR 3's earlier same-executable comparison, rather than the final corrected
executable checksum `34a404ebf976eed7b8b0a46d02902571da93357b45e41acafd5ecef53911ee9f`.
Do not claim that this local executable produced the final historical reports.
The historical corrected indexed-profile measurements are the control.

The [machine-readable evidence](mutation-range-quality.json) freezes both historical
per-mutant summaries and the final local PR 3 reports, tool versions, configuration
and header hashes, the new executable hash, independent run counts, a per-mutant
status matrix, exact boundary-failure output and debugger stacks. Only the known
checkout prefix is removed from IDs. Local raw reports retain opaque IDs verbatim.
All full-suite runs have exactly the same 47 IDs as PR 3: no added/missing mutants.
The deliberately smaller isolated selection discovers 40, recorded separately.
Local raw artifacts remain under `out/pr4`; committed evidence preserves the
per-mutant classifications without depending on ignored build files.

## Boundary tests

`resource_ranges_tests.cpp` adds two direct tests. The open-remainder test checks
an equal cut, positive cut and finite interval. The normalization test accepts an
end exactly equal to `UINT32_MAX`, requires a finite end, and rejects overflow.
Both pass on the original implementation. Mull's captured output proves:

| Original surviving ID suffix | Detecting assertion | Result |
| --- | --- | --- |
| `cxx_gt_to_ge:…:75:28:75:29` | `REQUIRE_FALSE(ambiguous_open_remainder(open, VkDeviceSize{128}))` | killed, `!true` |
| `cxx_gt_to_ge:…:114:39:114:40` | `REQUIRE(valid.has_value())` | killed, `false` |

No production-code mutation was needed to verify the failures; Mull executed the
actual instrumented variants. Exact failure output is in the evidence JSON.

## Timeout experiments

All runs use one worker, the same new instrumented executable, the same `mull.yml`,
`--timeout 10000`, `[ranges]`, and a freshly collected and merged indexed profile.
The initial matrix retains Catch2's default randomized order and continues after
failed tests. Every row below is three independent runs; K = killed, T = timed out.
Each ID is `mutator:include/vkexec/detail/resource_ranges.hpp:<coordinates>`.

| Mutator and coordinates | Default minimum | 250 ms | 1000 ms | 3000 ms |
| --- | --- | --- | --- | --- |
| `cxx_add_to_sub:35:81:35:82` | T/T/T | T/T/T | T/T/T | T/T/T |
| `cxx_lt_to_ge:40:33:40:34` | T/K/K | T/K/K | K/K/K | K/K/K |
| `cxx_eq_to_ne:142:35:142:37` | T/T/T | T/T/T | T/T/T | T/T/T |
| `cxx_ne_to_eq:161:71:161:73` | T/T/T | T/T/T | T/T/T | K/K/K |
| `cxx_lt_to_ge:167:25:167:26` | K/K/K | K/K/K | K/K/K | K/K/K |
| `cxx_lt_to_le:167:25:167:26` | T/T/T | T/T/T | K/K/K | K/K/K |
| `cxx_lt_to_ge:173:27:173:28` | K/K/K | K/K/K | K/K/K | K/K/K |
| `cxx_lt_to_le:173:27:173:28` | K/K/K | K/K/K | K/K/K | K/K/K |
| `cxx_add_to_sub:118:69:118:70` | K/T/K | K/T/T | T/K/K | K/K/T |

The first eight rows investigate every ID named in the PR plan. The saved final
PR 3 reports also contain a timeout at `118:69`; the union of those saved reports
has nine timeout IDs. This additional budget/order-sensitive result is preserved,
rather than discarding it to match the proposed eight-ID list.

| Minimum | Killed in three runs | Timed out in three runs |
| --- | --- | --- |
| Upstream default (30 ms floor; adaptive budget higher) | 42, 42, 43 | 5, 5, 4 |
| 250 ms | 42, 42, 42 | 5, 5, 5 |
| 1000 ms | 43, 44, 44 | 4, 3, 3 |
| 3000 ms | 45, 45, 44 | 2, 2, 3 |

There are zero survivors and zero uncovered in these runs. Increasing the budget
alone does not stabilize classification. The reported budget can exceed its floor:
for example the first default run's timeout durations were about 320 ms.

## Diagnosis and selected policy

Catch2 3.12's `ConfigData::runOrder` defaults to `Randomized` and generates a fresh
seed for each process. Thus unchanged source and Mull settings do not imply the
same test order. Captured logs show assertion failures before several timeouts.
GDB reproductions activate one exact mutant ID through its environment variable,
run `[ranges] --rng-seed 1`, interrupt after six seconds, and capture stacks:

* `35:81`: still processing the deterministic buffer reference test, step 191,
  in buffer conflict validation/overlap. Mutated normalization creates excessive
  region processing.
* `142:35`: deterministic image reference test, step 37, in
  `merge_image_regions` / `merge_compatible`; invalid intersections cause expensive
  fragmented-region processing.
* `118:69`: this fixed order reaches an empty-vector assertion in a later image
  test and aborts. Other orders time out. Treat it as order-sensitive broken
  execution, not a newly equivalent or uncovered mutant.

These stacks demonstrate excessive work, not a proof of mathematical
nontermination. No infinite loop is claimed and no production guard is added.
The direct algebra selection (buffer normalization, buffer interval algebra, image
rectangle subtraction) uses its own freshly collected indexed profile: 40 discovered,
35 killed, 5 survived, zero timed out. All nine timeout IDs above are killed there.
Those isolated survivors are outside this diagnostic question and do not replace
the full-suite result.

Fail-fast alone at 1000 ms gives 46/47/47 kills with 1/0/0 timeouts: random order
can still enter expensive work first. Explicit `--order lex --abort` gives 47 kills
and zero timeouts in all three runs at 1000 ms and all three at 250 ms. Lexical
order makes execution repeatable; fail-fast stops on a real Catch2 assertion before
subsequent tests process invalid mutated ranges. It does not convert a timeout
into a kill. All 47 mutants in each of the six final runs emit a Catch2 `FAILED:`
assertion; the committed evidence records every detecting assertion, independently
for each run. None of these final kills depends on a crash or signal.

Select `vkexec_MULL_WORKERS=1`, `vkexec_MULL_MINIMUM_TIMEOUT_MS=250`, and Catch2
`--order lex --abort` for both profile collection and mutant execution. The
250 ms floor is sufficient in repeated measurements with this execution policy;
1000 ms provides no observed classification benefit. This is a local measured
baseline, not a promise about arbitrary load or hardware. Timeouts remain separate
in all future reports. The raw-to-indexed profile generation, coverage selection,
score formula, mutator scope and production implementation remain unchanged.

## Final validation

Both actual CMake targets were run three times independently, preserving each
report/profile before the next target invocation:

| Audit | Total | Killed (each run) | Survived | Timed out | Uncovered | Score |
| --- | --- | --- | --- | --- | --- | --- |
| `ranges` × 3 | 47 | 47 | 0 | 0 | 0 | 100% |
| `ranges_uncovered` × 3 | 47 | 47 | 0 | 0 | 0 | 100% |

The 100% score applies to these 47 discovered mutations with the enabled
operators, resource-range source filter and `[ranges]` test selection. It does
not establish complete correctness or mutation coverage across vkexec. Other
audit scopes need their own measured test-order and timeout policies.

All 47 original IDs are discovered and evaluated; no unexpected coverage
regression. The unmodified `[ranges]` run passes all 55 cases and 72926 assertions.
The two Mull Python diagnostic CTest suites and their direct Python invocations
pass. Cache-input checks reject `0`, `-1`, `1.5`, and `invalid` for each new setting;
the valid configuration is restored afterward. The new settings are declared inside mutation setup only.
No score threshold, per-mutant exclusion, timeout-as-kill rule, or DAG expansion
is introduced.

The complete regular Clang debug configuration was explicitly reconfigured with
`-Dvkexec_ENABLE_MUTATION_TESTING=OFF` (the pre-existing debug directory had cached
it ON), rebuilt successfully, and passed CTest: 351 tests, three skipped GPU/queue
capability cases. The skips were the terminal DAG join, profiled DAG branches, and
cross-family profiling synchronization tests. The regenerated Ninja file has no
Mull frontend or LLVM profiling flags. Formatting checks and `git diff --check`
also pass.
