# Cancellation

A receiver may request stop through the sender/receiver protocol. If stop is observed before submission, the pass graph can complete as stopped without queueing work. Once work is submitted, vkexec still follows the fence completion path so resources are not released while the GPU may use them.

Cancellation completes through `set_stopped`, not `set_error`. A caller waiting synchronously should inspect the stopped outcome, or handle the disengaged optional in exception-enabled `sync_wait`. A stop request is not a promise that already-submitted Vulkan commands are preempted.

Keep the operation state, context, and borrowed resources alive until the terminal completion. See [the implementation error and cancellation model](../architecture/errors.md).
