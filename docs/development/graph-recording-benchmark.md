# External graph recording baseline

Measured on 2026-10-31 with Clang 22.1.8, C++20, Release (`-O3`), Linux,
and llvmpipe (LLVM 21.1.8, 256 bits). The host exposes 12 logical CPUs at
4400 MHz. CPU frequency scaling was enabled, so these are baseline samples,
not performance guarantees. Vulkan timestamps here measure the software device;
a discrete GPU baseline remains to be measured on suitable hardware.

The workload is a dynamic graph of identical single-workgroup compute passes.
Each pass increments a storage-buffer value using `recording.comp`. All graph
paths declare the same full-buffer read/write access. The manual path uses the
same `record_pass()` commands and resource-specific compute barriers. Every
path emits N-1 barriers. There is no production graph caching.

ManualRecording, OwnedRecorder, and ExternalRecorder exclude planning, command
buffer allocation/begin/end, queue submission, waiting, and cleanup from CPU
timing. OwnedRecorder calls the shared owned recorder with a prepared plan;
ExternalRecorder calls the external backend with the same prepared plan.
RecordInto measures the public API including validation, planning, ownership
transfer, and recording. Graph construction happens outside timing. GraphPlanning
isolates the shared planner. OwnedExecution executes the existing owned sender
through GPU completion; its wall time includes allocation, submission, and waits.

CPU times are microseconds per graph. Owned wall time is reported separately.

| Passes | Manual CPU | Owned recorder CPU | External recorder CPU | `record_into` CPU | Planning CPU | Owned execution CPU | Owned wall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 2.63 | 2.98 | 3.29 | 6.43 | 1.88 | 14.56 | 210.65 |
| 32 | 5.86 | 6.89 | 7.25 | 17.12 | 6.21 | 28.71 | 518.63 |
| 128 | 16.95 | 20.55 | 21.03 | 53.62 | 22.05 | 62.62 | 1852.71 |
| 512 | 45.46 | 59.35 | 63.92 | 215.31 | 115.26 | 228.72 | 6810.16 |

Vulkan timestamp durations in microseconds (submission and host waits excluded):

| Passes | Manual | Owned recorder | External recorder | `record_into` | Owned execution |
|---:|---:|---:|---:|---:|---:|
| 8 | 137.31 | 133.54 | 150.83 | 142.22 | 141.87 |
| 32 | 426.84 | 437.90 | 474.18 | 468.72 | 432.48 |
| 128 | 1635.07 | 1622.52 | 1755.41 | 1700.85 | 1715.13 |
| 512 | 6531.37 | 6405.31 | 6893.98 | 6680.65 | 6495.64 |

C++ heap allocations on the recording thread per graph:

| Passes | Manual / owned recorder / external recorder | Planning | `record_into` | Owned execution |
|---:|---:|---:|---:|---:|
| 8 | 0.00 | 32.00 | 33.00 | 38.00 |
| 32 | 0.00 | 106.00 | 107.00 | 112.00 |
| 128 | 0.00 | 396.00 | 397.00 | 402.01 |
| 512 | 0.00 | 1550.00 | 1551.00 | 1556.01 |

This instrumentation counts C++ `new` calls on the recording thread, including
those inside invoked libraries, but excludes driver `malloc` calls and other
threads. Descriptor allocations are zero for this fixture: one descriptor set is
bound before timing and shared identically by every mode. Temporary descriptor
lifetime is tested separately in the recording integration tests.

`recording_submissions` reports the number of successfully submitted scopes in
owned execution, derived from the shared execution plan rather than a one-per-graph
assumption. The linear fixture has one batch and therefore one submission.
`OwnedExecutionDag` uses the same passes as a chain of individual DAG batches;
it reports 8, 32, 128, or 512 submissions. Each fixture has one terminal, so no
extra join submission is needed. Recording/planning modes report zero submissions
inside their timed operations. For device timestamps, the harness submits each
recorded buffer once after CPU timing is paused. Dispatch-interception regression
coverage independently checks that the public API submits and waits zero times.

`record_pass()` returns `void`, matching Vulkan's command-recording functions;
there is no status to discard. The benchmark checks barrier, graph recording,
command-buffer finalization, submission, and completion results where available.

The owned and external recorders emit equivalent commands and have similar CPU
costs in this sample. At 512 passes their device timestamp durations differ by
about 7.6%; this short software-device run does not establish a hardware regression
threshold. Planning dominates much of the public API's additional CPU cost.

Reproduce after configuring with `-Dvkexec_BUILD_BENCHMARKS=ON`:

```sh
cmake --build build --target vkexec_graph_recording_benchmark
build/benchmark/vkexec_graph_recording_benchmark \
  --benchmark_min_time=0.02s \
  --benchmark_out=recording.json --benchmark_out_format=json
```

The benchmark reports timestamp durations only when the selected queue supports
timestamps. Build and run the recording tests with synchronization validation:

```sh
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_VALIDATION_VALIDATE_SYNC=1 \
  build/test/vkexec_core_tests '[record_into]'
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_VALIDATION_VALIDATE_SYNC=1 \
  build/test/vkexec_profiling_tests '[record_into]'
```

Validation on this configuration completed 321 CTest cases with no failures
and 8 capability skips. Recording tests passed 543 assertions across 13 cases
with synchronization validation enabled; the external profiling test also passed.

Verify the environment setting with a positive control before trusting a clean run:

```sh
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
VK_VALIDATION_VALIDATE_SYNC=1 VK_VALIDATION_SYNCVAL_RECORD_TIME_VALIDATION=1 \
  build/test/vkexec_sync_validation_probe
```

The probe must print `SYNCVAL_PROBE_CONFIRMED` and exit zero. It deliberately
records an unsynchronized WAW hazard, captures a `SYNC-HAZARD` diagnostic, and
resets the command buffer without submitting it. Failure to observe the hazard
is an error, including when the validation layer or setting is unavailable.
With `vkexec_TEST_VALIDATION=ON`, CTest also runs this positive control. The
[upstream SyncVal instructions](https://chromium.googlesource.com/external/github.com/KhronosGroup/Vulkan-ValidationLayers/+/HEAD/docs/syncval_usage.md)
document the `VK_VALIDATION_VALIDATE_SYNC` setting.
