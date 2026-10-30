# GPU graph execution mutation experiment (PR 12)

This experiment is opt-in and noncanonical. It asks whether Vulkan execution
assertions detect host-side defects in `pass_execution.hpp` missed by the CPU
DAG tests. The five canonical scopes, ten public audit targets, preset and
baseline files are unchanged. No GPU audit runs in GitHub Actions.

## Current findings

Hardware measurement is **inconclusive**. This environment exposes no `/dev/dri`.
The compiled strict preflight selected llvmpipe, reported a CPU device, and exited
86 with `GPU_INFRASTRUCTURE_FAILURE: software device rejected; hardware measurement inconclusive`.
No hardware mutation audits were started and no mutation counts or scores are
reported. A separate `vulkaninfo --summary` invocation also failed during driver
enumeration; this is infrastructure evidence, not a detected mutant.

With an explicit software diagnostic override, the preflight created a context,
VMA allocator, mapped staging allocation and device-local allocation. Both GPU
tests then passed three times without skips on llvmpipe. Each test submitted real
Vulkan transfers and checked every output word. These software diagnostic runs
used the existing debug build's libraries and the new sources compiled with
Clang 22.1.8 and warnings as errors. They were not isolated, committed-source
Mull measurements. They establish that the fixtures execute, not that a physical
GPU is eligible or that mutation testing provides incremental fault detection.

The machine-readable [snapshot](mutation-experiments/gpu_graph_execution.json)
records this distinction, device properties, exact tool versions and evidence
hashes. Local logs and JUnit reports are in `out/pr12-evidence/gpu-initial/`.
Hardware repeatability, mutant emission, CPU/GPU overlap, survivor analysis and
incremental detections remain unmeasured.

**Recommendation:** keep GPU mutation testing opt-in. There is no demonstrated
incremental value supporting permanent GPU mutation infrastructure or a scheduled
GPU runner. Run the bounded three-repeat experiment on controlled hardware before
making that decision. This PR does not establish completion of the measurement
acceptance criteria. If hardware results detect only already-killed CPU mutants,
retain useful Vulkan tests and stop expanding mutation infrastructure.

## Fixtures and device policy

`test/mutation/gpu/graph_execution_tests.cpp` reuses the resource-region overlap
scenario: two initial fills, an overlapping snapshot, a partial overwrite, final
copy and host barrier. Every snapshot/final word is checked. The disjoint test
writes each half separately and copies both downstream, checking all words,
including endpoints. Both use the context's compute queue; Vulkan compute queues
support transfer operations. There are no shaders, optional image formats or
cross-family ownership requirements.

`strict_gpu.hpp` uses successful sender outcomes instead of the integration
helpers' permissive skips. Context/allocation/submission/invalidation failures
emit a distinctive infrastructure marker and exit 86. Assertion failures use
Catch2. The selected physical device's device UUID, driver UUID, vendor/device
IDs, API and driver versions form its identity. The runner pins this identity
in the environment; each test-created context checks it. Preflight records the
selected device even when it rejects software, verifies enumeration and a usable
compute queue, and exercises both memory domains. Loader/device/allocator failures
fail the experiment. CPU devices require `--allow-software` and are diagnostic
only. Use `VK_DRIVER_FILES` (or legacy `VK_ICD_FILENAMES`) to constrain device
selection; environment selections are retained. On systems with several devices,
check `device.json` before accepting results.

## Reproduction on hardware

Commit the implementation first. The runner refuses any tracked or untracked
source changes and refuses to overwrite evidence directories. Enter the pinned
shell and run serially from the repository root:

```sh
nix develop .#clang-mut
python3 scripts/gpu_mutation_experiment.py --artifacts out/pr12-evidence/gpu-run-1
python3 scripts/gpu_mutation_experiment.py --artifacts out/pr12-evidence/gpu-run-2
python3 scripts/gpu_mutation_experiment.py --artifacts out/pr12-evidence/gpu-run-3
python3 scripts/gpu_mutation_experiment.py --compare \
  out/pr12-evidence/gpu-run-1/experiment.json \
  out/pr12-evidence/gpu-run-2/experiment.json \
  out/pr12-evidence/gpu-run-3/experiment.json
```

For an explicitly separate software diagnostic, pass `--allow-software` and use
a fresh evidence directory. Software and hardware measurements are incomparable.
No public preset is added. The wrapper checks Clang/LLVM 22.1.8 and Mull 0.35.0,
creates its own detached temporary worktree, and overlays only `CMakeLists.txt`
and `test/CMakeLists.txt`. The complete patch and overlay hashes are retained.
The extra source, preflight target and sixth scope exist only in that worktree.
The experiment configuration is byte-identical to the DAG configuration:
`cxx_comparison` and `cxx_boundary`, restricted to `pass_execution.hpp`.

The existing preset is configured with VMA on, tools off, one Mull worker,
60,000 ms baseline timeout and 5,000 ms minimum adaptive mutant timeout. These
are experimental values. Review the recorded baseline times before retaining
them. A separate 1,800 second process-group timeout bounds each command; override
it with `--outer-timeout`. The runner explicitly configures the existing private
scope build from the parent's generated cache, builds the binary and preflight,
checks three JUnit baselines, then invokes the covered and inclusive public
scope audit targets serially. It does not call `scripts/ci/run_mutation_audits.py`.
Each audit collects and merges its own LLVM profile through the existing CMake
mechanism. Preflight and an unmutated test run follow each audit, including an
audit that failed or timed out.

Only the worktree created by this invocation is force-removed. Partial SQLite,
JSON and profiles are copied in an audit's `finally` block before health checks
or cleanup. Logs, command arguments, build cache, source/configuration hashes,
versions, binary hash, environment selections, wall times and explicit failure
states are retained outside the worktree. Failed cleanup is itself a failure.

## Reviewing results

The runner rejects unknown source files, operators or statuses, duplicate
normalized IDs, missing/empty profiles, invalid SQLite and JSON contradictions.
Covered IDs and statuses must exactly equal inclusive rows excluding
`not_covered`. An empty covered report is permissible only when inclusive has
no evaluated rows; it is not evidence of fault detection.

`comparison.json` joins the CPU DAG baseline and GPU report on exact normalized
IDs. It reports the intersection, GPU-only and CPU-only populations separately;
it never requires all 57 CPU DAG mutants to appear in another executable. Raw
Mull kills of CPU survivors or uncovered mutants are candidate incremental
findings, not confirmed detections. Killed-by-both overlap adds no incremental
value. GPU-uncovered rows are unevaluated; CPU-killed/GPU-survived rows indicate
a weaker GPU oracle for that specific mutant.

A Mull `killed` status can include a failing host/Vulkan operation. Every raw
kill remains explicitly **unverified** until a per-mutant replay distinguishes
an observable Catch2 assertion from device loss, allocation/submission/readback
failure or driver instability. Infrastructure markers in available audit output
invalidate the measurement. Mull can suppress child output, so absence of the
marker does not establish assertion attribution. Do not count unreviewed kills
as additional fault detections. Timeouts, crashes and abnormal exits retain
their original statuses. No mutation score is produced automatically.

For uncovered rows the runner exports LLVM coverage and applies the PR 10
`diagnose_mull_coverage.py` analysis. Inspect `attribution.json` and the export
before adding assertions. For full switch-gap attribution, retain Mull
`--debug --debug-coverage` output using the same binary and indexed profile;
the ordinary public audit log may lack the ranges needed to identify the known
Mull switch-gap limitation. Preserve this diagnostic evidence separately from
the initial measurement. Add assertions only for deterministic observable
behavior, and retain both initial and final snapshots.

Three measured manifests must have the same revision, versions, selected device,
source/configuration/overlay fingerprints and timeout policy. Comparison lists
any ID whose population membership or status varies, including killed/survived/
timed_out/crashed changes. Unstable results have no passing score. Stable raw
kills still require assertion attribution. Any future permanent GPU automation
requires controlled hardware plus demonstrated, repeatable incremental detections.

## Lightweight validation

```sh
python3 -m unittest discover -s test/mutation/gpu_experiment -v
```

These tests need neither Vulkan nor the pinned toolchain. They exercise source
cleanliness, exact overlays, unexpected changes, device policy, baseline skips,
SQLite validation, exact-ID comparisons, differing device fingerprints,
process-group termination, partial artifact retention, full orchestration and
worktree cleanup. The mutation workflow runs them for relevant pull requests;
its Ubuntu audit job continues to run only the canonical CPU scopes.
