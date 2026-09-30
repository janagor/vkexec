# Completion

A successful GPU pass `start()` submits asynchronously. The operation state and borrowed resources must remain alive until a terminal receiver completion. The GPU fence marks execution complete; the completion agent observes it and transfers value completion to the context host scheduler.

Post-GPU callbacks run after the fence and before downstream `set_value`. This ordering makes it safe to perform cleanup that must wait for GPU use to end. It does not make pass construction or `start()` a blocking call.

`vkexec::sync_wait` is an explicit blocking consumer that always returns an optional value tuple. Sender errors throw `std::system_error` when exceptions are available and terminate otherwise. `vkexec::try_sync_wait` returns an outcome so callers can inspect value, error, or stopped state. See [Errors](errors.md) for the three completion channels.
