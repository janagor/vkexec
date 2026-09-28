# Execution model

vkexec is a stdexec sender domain for Vulkan work. Scheduling on a context gives the pipeline its execution environment; pass adaptors add semantic GPU operations. The resulting sender is a description of work, not a recorded command buffer.

```text
CPU scheduler -> semantic sender pipeline -> Vulkan queue
    -> GPU execution -> completion agent -> context host scheduler
    -> downstream receiver
```

## From description to submission

Composing `stdexec::schedule(ctx->get_scheduler()) | vkexec::compute_pass(...)` retains the pass arguments and borrowed resource references. `connect()` produces operation state, but still does not submit. `start()` checks for an early stop, records a command buffer, submits it, and returns without waiting when submission succeeds. Multiple statically composed passes can be collected into one typed pass graph.

## GPU completion and host continuation

The completion agent observes the submission fence. Once GPU work finishes, value completion moves to the context host scheduler. Any `after_gpu` actions run there before the downstream receiver sees `set_value`. A pass-local GPU barrier orders memory accesses within a graph; it is separate from this host completion step.

Use `vkexec::sync_wait` when the current thread must see results before continuing, or connect the sender to a receiver for asynchronous completion. [The first program](../getting-started/first-program.md) demonstrates the blocking boundary. [Execution architecture](../architecture/execution.md) describes the implementation path.
