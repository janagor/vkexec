# Graph submission experiment (PR4)

Base: PR2, `aa9154f` (`perf/graph-execution-optimization`). PR3 was not incorporated.

No compilation, formatter, linter, test, validation, or benchmark execution was
performed: the session explicitly prohibited compilation, formatting, and lint
checks. No PR4 result was collected. Do not interpret the implementation or
structural counters as performance acceptance.

## Measurements to collect

Use equivalent Release builds and the same device, driver, feature configuration,
and host load for PR2 and PR4. Capture benchmark context and repetitions:

```
vkexec_graph_submission_benchmark --benchmark_repetitions=7 --benchmark_report_aggregates_only=true --benchmark_out=graph-submission-pr4.json --benchmark_out_format=json
```

Use the existing PR2 results for comparison. The PR2 edge-scanning reference
is retained only for correctness tests.

The CPU groups cover 8, 32, 128, and 512 batches in binary and timeline modes.
Topology arguments are linear, independent, fork/join, dense, fan-in, and
alternating queue-family metadata, in that order. Plans are built outside the
CPU timing loop. These fixtures model dependency edges, including fan-in; they
do not measure resource-use mapping. The mixed-queue CPU fixture checks that
queue metadata does not alter dependency mapping; it does not execute real
cross-queue ownership transfers.

`bm_graph_execution` records empty real command buffers and runs the production
submission sender through asynchronous terminal completion and cleanup, after
one warm-up graph. It includes fixture construction, recording, synchronization
creation, preparation, submission, and completion. It adds the same terminal
join required by production when independent sinks exist. Both GPU benchmarks
use the same fixture builder for the final plan and recording. It does not measure
public pass/resource planning or shader work. GPU mode follows the device's
actual timeline support; the third argument is not a forced feature override.

`bm_submission_stages` completes one untimed warm-up graph through the same
execution helper as `bm_graph_execution`, then reports mean wall seconds per iteration for object
creation, preparation, and queue submission. Recording and fence waiting are
excluded from those stage counters. Counter collection and stage accumulation
happen while timing is paused; counters are published after the loop. Fixture
destruction also completes before timing resumes. Each submitted fence is retired before
cleanup, including failures. The fence/semaphore/submit counters describe this
fixture's per-iteration operations. Submission call time includes whatever
`context::submit()` does internally. GPU completion time is never charged to
CPU preparation. Object creation is grouped, so distinguishing individual
`vkCreateFence` and `vkCreateSemaphore` costs still requires profiling.

The 512-batch GPU cases require `VKEXEC_BENCH_EXPENSIVE_GPU=1`. Mixed-queue GPU
cases alternate between compute and graphics queue handles and skip explicitly
when a distinct graphics queue is unavailable. Both benchmarks use the default
compute-only `factory::make_context({})`; the GPU test uses `require_context()`
with the same default queue setup. These fixtures do not request or adopt a
second queue, so multi-queue GPU coverage remains unavailable in the default
compute-only setup. A successful suite run with these cases skipped does not
validate cross-queue synchronization. Actual two-queue GPU validation requires
a dedicated context fixture with two real queue handles; it remains pending.
The full GPU path uses actual capability detection; forced binary-mode GPU
validation remains pending. Temporary allocation counts are not instrumented;
there is no allocation counter claiming measured allocations. Inspection shows
that `submit_batch()` itself no longer constructs vectors or scans edges.

## Validation and acceptance still required

The new CPU test compares exact wait/signal ordering, handles, values, and stage
masks to the PR2 reference for empty, single, independent, chain, fork/join,
dense, fan-in, and mixed-family plans, with and without presentation entries,
in binary and timeline modes. Existing planner, presentation, cancellation,
fault-injection, and partial-submission retirement tests remain relevant and
must be run, along with Vulkan synchronization validation. The new test does
not replace family ownership or lifecycle integration coverage. An additional
GPU graph test exercises source fan-out and a join across distinct compute and
graphics queue handles, skipping when two handles are unavailable. It uses empty
command buffers and does not verify resource ownership transfers.

Inspect repeated CPU and complete-execution results before accepting the change:
look for >=20% preparation improvement on edge-heavy plans and investigate
reproducible complete-execution regressions above approximately 5%. No speedup,
regression bound, or correctness pass is claimed yet. Fence counts, semaphore
ownership policy, queue assignments, and execution planning are unchanged.
