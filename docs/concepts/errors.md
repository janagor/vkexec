# Errors

vkexec sender operations have three distinct terminal channels. `set_value` reports successful completion after promised GPU work and cleanup. `set_error(vkexec::error)` reports validation, recording, submission, completion, or cleanup failure. `set_stopped` reports cancellation.

`vkexec::sync_wait` always returns an optional value tuple. It throws `std::system_error` on sender error when exceptions are available and terminates otherwise. `vkexec::try_sync_wait` returns sender error completion as an outcome, allowing a caller to inspect `failed()`, `stopped`, and `values` explicitly. When C++ exceptions are enabled, unrelated exceptions from sender setup such as `connect()` may still propagate. A failure before queue submission need not wait for the GPU.

Do not interpret a stopped operation as a Vulkan error or read output buffers before a successful completion. [The first program](../getting-started/first-program.md) checks a completed result; [architecture details](../architecture/errors.md) describe exception containment in operation state.
