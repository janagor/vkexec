# Recording a graph into a caller command buffer

Include `<vkexec/record_into.hpp>` and pass an already recording **primary**
command buffer, a graph, and a queue belonging to the context:

```cpp
auto graph = stdexec::schedule(ctx.get_scheduler())
  | vkexec::compute_pass(producer_bind, producer_params, dispatch)
  | vkexec::compute_pass(consumer_bind, consumer_params, dispatch);
auto recorded = vkexec::record_into(ctx, cmd, std::move(graph),
  { .queue = ctx.compute_queue_ref() });
if (!recorded) {
  // A recording failure can leave partial commands. Discard/reset cmd;
  // do not end and submit it or continue recording usable work into it.
  return vkexec::fail(recorded);
}
// Additional caller commands may follow, with caller-supplied dependencies.
vkEndCommandBuffer(cmd);
// Submit cmd using the caller's submission mechanism.
// After successful submission (including synchronous submit-and-wait):
auto marked = recorded->mark_submitted();
if (!marked) return vkexec::fail(marked);
// Keep recorded alive until completion has been established by the caller.
// After the caller's fence/timeline/completion mechanism confirms completion:
auto retired = recorded->complete();
```

Static expressions, materialized static graphs, and consumed dynamic graphs are
supported. Copyable static lvalues are copied; rvalues transfer their steps.
Move-only dynamic steps are retained without copying. A scheduled root must
belong to `ctx`; arbitrary host sender execution is not part of this synchronous API.

`record_into` shares the owned graph execution planner, resource tracker, and
per-step barrier recorder. It never allocates, begins, ends, resets, or submits
command buffers and never allocates submission synchronization or waits for GPU
completion. Passes can allocate their own temporary descriptors or profiling
queries. Custom pass callbacks retain their ordinary responsibilities.

All resolved graph queues must match the supplied queue by **both handle and
family**. DAG nodes run in their existing topological recording order, including
independent branches. Cross-queue work and attached presentation are rejected
before any graph commands are emitted. The command buffer's family must be
compatible with the supplied queue. It must be in a legal recording context for
every pass (for example, compute dispatches must be outside rendering). Vulkan
cannot query these recording-state preconditions from a raw command buffer.

The caller owns entry and exit dependencies. Image initial layouts must match
actual layouts on entry. Generated resource barriers cover declared graph
accesses only. There are no implicit presentation transitions and no persistent
resource state between calls. Keep referenced resources, pipelines, and other
borrowed pass data alive through completion. Temporary descriptor sets and the
step instances themselves are retained by the returned `recorded_graph`.

After successful submission, call `mark_submitted()` before any retirement.
This transition is required even if a synchronous submit-and-wait has already
returned. Call `complete()` only after confirmed GPU completion. Callbacks run synchronously
on the calling thread. It runs `after_gpu()` and
then `commit_after_gpu()`, once, before freeing temporary descriptors and steps.
If a callback throws, the object is retired, cleanup runs, and `complete()`
returns an error; callbacks are not retried. If recording succeeds but the work
will never be submitted, first discard/reset the command buffer, then call
`abandon()`. Abandon frees recording resources without successful-completion
callbacks. `complete()` rejects a recording that has not been marked submitted;
`abandon()` rejects a submitted recording. If submission fails without accepting
work, discard/reset the buffer and abandon the still-unsubmitted recording.
Vulkan submission and completion cannot be inferred by this object: the caller
must report the submission transition and confirm completion. Forgetting to report
submission still violates the `abandon()` precondition; the library cannot inspect
a raw command buffer to detect that mistake.

Repeated retirement returns `invalid_argument`. Move construction and assignment
transfer ownership; an assignment destination must already be retired or empty.
Destroying or overwriting an active recording asserts in debug builds and
terminates in release builds, without waiting or releasing potentially in-flight
resources. An empty graph still returns a retirement object with this contract.

If recording fails, no submission occurs. Temporary recording allocations are
released, which can invalidate the unsubmitted command buffer. The error result
requires the caller to discard/reset that buffer before reuse. No GPU use of the
failed commands is permitted. This rule also applies to recording exceptions.
