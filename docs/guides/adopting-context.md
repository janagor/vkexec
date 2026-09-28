# Adopting a context

Use `factory::adopt_context` when your application already owns a Vulkan instance, physical device, logical device, and compute queue. The factory returns a sender completing with a `std::unique_ptr<context>`. Starting it sets up vkexec's command pool and host/completion agents around the borrowed objects.

## Supply matching handles and capabilities

`context_adopt_info` requires the instance, physical device, device, compute queue, and compute queue family. Set `api_version` to the version negotiated by the embedder. If synchronization2 is enabled, report the feature and any KHR-path requirements accurately; vkexec uses this information for barrier lowering. Graphics and present queues are optional when the application only runs compute work.

```{literalinclude} ../../include/vkexec/context.hpp
:language: cpp
:start-at: struct context_adopt_info
:end-at: };
```

The snippet shows the public fields; the compiled context tests and your Vulkan device setup determine the values. The embedder remains responsible for queue family correctness and for keeping the borrowed handles alive until every vkexec operation and the context have finished.

## Keep allocation separate

Adopting a context does **not** adopt or create a VMA allocator. You can continue using your own Vulkan memory system because pass and binding APIs consume borrowed Vulkan handles. If you want vkexec's VMA wrappers around an existing `VmaAllocator`, call `vma::factory::adopt_allocator(*ctx, external_vma)` separately. It borrows the allocator; vkexec does not destroy the external VMA handle.

Finish GPU work, destroy context-dependent resources and the vkexec context, then destroy the embedder's device and instance. See [Ownership](../concepts/ownership.md) for the full lifetime rule.
