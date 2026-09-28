# Error and cancellation implementation

## `vkexec::error`

Expected Vulkan and vkexec failures are normalized into the `vkexec::error` completion channel.

## `set_value`

Successful operations complete through `set_value` only after their promised work and required post-GPU cleanup are done.

## `set_error`

Validation, recording, submission, completion, or cleanup failures complete through `set_error`. A failure before submission need not wait for the GPU.

## `set_stopped`

Cancellation completes through `set_stopped`; it is not an error value. Stop checks occur before work begins where the sender protocol permits them.

## Exception containment

Operation `start()` paths remain `noexcept`. Potentially throwing internals are caught when exception support is enabled and translated to the error channel. Exceptions must not escape through a receiver completion path.

## Synchronous boundaries

Blocking consumers such as `sync_wait` are explicit synchronous boundaries. They do not change the asynchronous contract of a successfully submitted GPU pass sender.
