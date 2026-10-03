# Contexts and schedulers

A `context` coordinates Vulkan device and queues, command pool, submission synchronization, and host and completion agents. `factory::make_context()` creates an owned instance and device. `factory::adopt_context()` wraps an application's existing Vulkan objects without taking their destruction ownership; the caller reports enabled capabilities and valid queue families.

A `scheduler` retains the context runtime independently of the public `context` object. `stdexec::schedule(ctx->get_scheduler())` creates a sender and carries that runtime into subsequent GPU operations. Scheduling alone does not record commands or dispatch to the GPU. Senders, connected operations, pass graphs, and submit scopes can finish after the public context is destroyed. For adopted contexts, the embedder must keep its Vulkan handles alive until all users of the runtime have been released.

Resources that require an explicit `destroy(ctx, ...)` still need the public context for cleanup. Borrowed pipelines, buffers, and other resources must remain valid through submitted work. Adopting a context does not adopt an allocator. For the embedding path, read [Adopting a context](../guides/adopting-context.md); for ownership rules, read [Ownership](ownership.md).
