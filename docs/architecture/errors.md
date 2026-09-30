# Error and cancellation implementation

## `vkexec::error`

Expected Vulkan and vkexec failures are normalized into the `vkexec::error` completion channel.
Synchronous helpers return `result<T>` or `status`; senders complete with `set_error(vkexec::error)`. `error::code` uses `std::error_code`. `make_vk_error_code` accepts failed (negative) `VkResult` values only; nonnegative Vulkan status values remain ordinary operation results.

The two categories are singletons in the linked vkexec library. Independently linked copies in separate shared objects do not have equal `std::error_category` identities. Boost's former category IDs provided that guarantee; this API no longer promises it.

## `set_value`

Successful operations complete through `set_value` only after their promised work and required post-GPU cleanup are done.

## `set_error`

Validation, recording, submission, completion, or cleanup failures complete through `set_error`. A failure before submission need not wait for the GPU.

## `set_stopped`

Cancellation completes through `set_stopped`; it is not an error value. Stop checks occur before work begins where the sender protocol permits them.

## Exception containment

Operation `start()` paths remain `noexcept`. Potentially throwing internals are caught when exception support is enabled and translated to the error channel. Exceptions must not escape through a receiver completion path.
Unexpected exceptions become `errc::unexpected_exception`. Allocation failures in builds without C++ exceptions are unrecoverable.

## Synchronous boundaries

Blocking consumers such as `sync_wait` are explicit synchronous boundaries. They do not change the asynchronous contract of a successfully submitted GPU pass sender.
`try_sync_wait` returns sender value, error, or stopped completion in both compiler modes. It converts sender error completion into an outcome rather than an exception. When C++ exceptions are enabled, exceptions from generic sender setup such as `connect()` may still propagate. `sync_wait` always returns an optional tuple: sender error completion throws `std::system_error` when exceptions are available and terminates otherwise; stopped returns an empty optional. Unrelated exceptions from sender setup are not converted into `std::system_error`. `sync_wait_value` uses the same throwing adapter and treats stopped as `errc::cancelled`.
