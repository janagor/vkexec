# Errors

vkexec sender operations have three distinct terminal channels. `set_value` reports successful completion after promised GPU work and cleanup. `set_error(vkexec::error)` reports validation, recording, submission, completion, or cleanup failure. `set_stopped` reports cancellation.

The synchronous public boundary depends on exception configuration. `vkexec::sync_wait` throws on sender error in exception-enabled builds and returns an outcome in no-exception builds. `vkexec::try_sync_wait` always returns a nonthrowing outcome, allowing a caller to inspect `failed()`, `stopped`, and `values` explicitly. A failure before queue submission need not wait for the GPU.

Do not interpret a stopped operation as a Vulkan error or read output buffers before a successful completion. [The first program](../getting-started/first-program.md) checks a completed result; [architecture details](../architecture/errors.md) describe exception containment in operation state.
