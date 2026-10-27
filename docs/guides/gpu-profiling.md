# GPU profiling

Configure with `-Dvkexec_BUILD_PROFILING=ON` and link `vkexec::profiling`.
The profiling target is optional; core `vkexec::vkexec` does not depend on it.

## Pass graphs

Wrap one primitive pass adaptor to measure its GPU commands:

```cpp
#include <vkexec_profiling/profiling.hpp>

auto profiler = vkexec::profiling::profiler::create(ctx);
auto graph = stdexec::schedule(ctx.get_scheduler())
  | vkexec::profiling::profiled(*profiler, "first", vkexec::compute_pass(first_bind, first_groups))
  | vkexec::profiling::profiled(*profiler, "second", vkexec::compute_pass(second_bind, second_groups));
auto completed = vkexec::sync_wait(std::move(graph));
auto events = profiler->take_completed();
```

The wrapper retains the pass's resource declarations and queue affinity. It
records timestamps inside the graph's existing command buffer and publishes
events after GPU completion and all pass callbacks succeed. `start()` remains asynchronous. The
measured interval begins after graph-generated barriers before the pass and
ends before graph-generated barriers after it. Barriers recorded by the pass
itself are included. The wrapper accepts one primitive pass; it does not span
composite senders or multiple submissions. Unsupported queue families and
query capacity failures are reported as graph recording errors. Graph errors
and cancellation do not publish successful samples. Copying an unstarted graph
with copyable passes creates a separate operation; each execution reserves its
own query pairs. A profiler belongs to the Vulkan device used to create it;
recording a profiled graph on another device returns an error before reserving
queries.
The wrapper resets queries before recording the primitive pass, so the pass
must begin outside an active render pass or dynamic rendering scope. It may
begin and end its own rendering scope internally.

## Manual command buffers

```cpp
#include <vkexec_profiling/manual_capture.hpp>

auto profiler = vkexec::profiling::profiler::create(ctx);
auto capture = vkexec::profiling::manual_capture::create(
  *profiler, ctx, command_buffer, queue);
{
  auto region = capture->scoped_region("dispatch");
  vkCmdDispatch(command_buffer, group_count_x, group_count_y, group_count_z);
}
// End recording, submit with a fence, and check submission success.
capture->mark_submitted(fence);
// After the fence signals:
capture->resolve_after_gpu();
auto events = profiler->take_completed();
```

The caller supplies the actual queue family and a fence for the command buffer's
submission. `resolve_after_gpu()` checks the fence without waiting and returns
an error while the GPU is still working. The fence must remain valid and must
not be reset or reused until resolution. The capture must also remain alive through resolution; destroying
a submitted unresolved capture is a contract violation because its query slots
could still be in use by the GPU. An unsubmitted capture can be discarded.
Call `mark_submitted()` immediately after a successful queue submission, before
any operation that may throw or return early. If submission succeeds, retaining
the capture and fence until resolution is required even when later work fails.
After a successful fence check, `discard_after_gpu()` can release the slots
without reading results. `VK_ERROR_DEVICE_LOST` makes the capture terminal and
quarantines its query slots. Other Vulkan fence errors leave the capture and
query slots intact so the caller can retry once the error is handled. A
terminal error is returned without terminating the process. Quarantined slots
are never recycled by that profiler.

Each capture is for one recording and one submission. Once it is resolved or
discarded, do not submit that recorded command buffer again without re-recording
its timestamp commands under a new capture. Discarding an unsubmitted capture
requires resetting or abandoning its command buffer before any submission.

Regions may nest, but creation and destruction must occur while recording
outside render passes and video coding scopes. Do not move a capture with live
regions. The timestamp commands measure GPU execution, not CPU recording time.
The start uses `TOP_OF_PIPE` and the end uses `BOTTOM_OF_PIPE`.

`profiler::supported(queue)` reports support for both timestamps and the command
buffer query reset strategy. Transfer-only queues are rejected.
Samples with unavailable queries or readback failures carry a `sample_status`;
they do not acquire an invented duration. `take_completed()` removes collected
events. Once `max_completed_events` is reached, later events are dropped and
`dropped_events()` counts them. Query slots are reused after resolution or
abandonment of an unsubmitted capture. A region lasting longer than the full
period of its queue family's valid timestamp bits has an ambiguous duration.

The profiler retains the context runtime. For adopted contexts, the embedder
must keep the borrowed Vulkan device and physical device alive until profiler
and capture handles are destroyed.

## Chrome Trace export

`write_chrome_trace()` accepts completed events and writes Chrome Trace JSON.
It returns the number of samples omitted because their status was not `valid`.
Export one submission at a time; mixed submission IDs or queue families return
an error because their timestamp domains have not been calibrated. The trace
uses the first recorded region as time zero and preserves gaps between regions
within that submission. Timestamp counter wrap is handled using the queue
family's valid bit width, but an interval spanning a full counter period is
ambiguous.

```cpp
#include <vkexec_profiling/chrome_trace.hpp>
#include <fstream>

auto events = profiler.take_completed();
std::ofstream output("gpu-trace.json");
auto exported = vkexec::profiling::write_chrome_trace(output, events);
if (!exported) {
  // Handle export error.
}
```

The `gpu_profiling` example is built when profiling and tools are enabled. It writes
one `gpu-trace-<submission-id>.json` file per submission, runs two compute
dispatches, and prints their measured GPU
durations before writing the trace. Event names must be valid UTF-8; the
exporter rejects malformed names before writing output. Submission IDs are
allocated by one process-wide counter shared by graph and manual captures.
Within a submission, region indices are assigned when recording begins and
identify the first recorded region used as the trace origin.
