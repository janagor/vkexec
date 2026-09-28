# Execution model implementation

```{graphviz} diagrams/lifecycle.dot
```

## `context`

`context` coordinates the Vulkan device and queues, command pool, submission synchronization, host agent, and completion agent. It owns its internal machinery. `factory::make_context()` creates an owned instance/device; `factory::adopt_context()` borrows an embedder's handles and queues. A context is an execution resource, not itself a stdexec scheduler.

## `scheduler`

`vkexec::scheduler` is associated with a context. `ex::schedule(ctx->get_scheduler())` produces a sender whose environment advertises both the vkexec domain and the context scheduler for value completion. Its value completion runs on the context's host agent. `schedule()` does not execute GPU commands; the sender environment carries the context information used for lowering subsequent GPU operations.

## Scheduler environments

`scheduler_env` advertises the vkexec domain and the context scheduler for `set_value`. This is a value-completion guarantee; error and stopped completions need not run on that scheduler. A domain-only environment can advertise vkexec lowering without claiming a completion scheduler. Pass adaptors require a predecessor whose value completion is guaranteed on `vkexec::scheduler`.

## Semantic expression lowering

Semantic expressions are normalized into typed pass steps before pass-graph materialization. See [Domain lowering](domain-lowering.md) for the domain transform and composite-operation rules.

## Pass graphs

A pass graph owns its ordered steps. `connect()` creates an operation state containing those steps and the receiver. It still does not submit work.

## Submission and completion

On `start()`, the operation checks for a requested stop, opens and records a command buffer, then starts the fence-backed submission sender. The successful path submits and returns without waiting for GPU completion. After the fence signals, `ex::continues_on(fence_sender, ctx->get_scheduler())` transfers completion to the context host scheduler. `after_gpu` callbacks run there before the downstream receiver completes. The completion agent observes GPU progress; it is not an arbitrary executor for downstream user code.

`start() != wait for GPU`. `sync_wait(...)` is an explicit blocking boundary.

## Cancellation and completion channels

The receiver observes `set_value` on success, `set_error(vkexec::error)` on failure, and `set_stopped` on cancellation. A stop requested before submission can prevent work from being submitted. Submitted work still follows the fence completion path; cancellation is not an error.
