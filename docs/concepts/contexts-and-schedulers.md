# Contexts and schedulers

A `context` coordinates Vulkan device and queues, command pool, submission synchronization, and host and completion agents. `factory::make_context()` creates an owned instance and device. `factory::adopt_context()` wraps an application's existing Vulkan objects without taking their destruction ownership; the caller reports enabled capabilities and valid queue families.

A `scheduler` is associated with a context but is a separate stdexec object. `stdexec::schedule(ctx->get_scheduler())` creates a sender and carries context information into subsequent GPU operations. Scheduling alone does not record commands or dispatch to the GPU.

The context must outlive connected operations, submitted work, and resources whose cleanup depends on it. Adopting a context does not adopt an allocator. For the embedding path, read [Adopting a context](../guides/adopting-context.md); for ownership rules, read [Ownership](ownership.md).
