# Resource and ownership model

```{graphviz} diagrams/ownership.dot
```

## Borrowed execution resources

Core execution consumes Vulkan handles such as `VkBuffer`, `VkImage`, pipelines, layouts, and descriptor sets. Recording does not ask a concrete allocator for memory. Borrowed resources must remain valid through GPU completion and any `after_gpu` work that uses them.

## `handles::`

`handles::` types group borrowed Vulkan handles and associated metadata for execution. They do not transfer ownership of the underlying Vulkan objects.

## `owned::`

`owned::` types are RAII wrappers where vkexec provides ownership. Their destruction requirements are separate from the borrowed handle types passed to execution algorithms.

## Allocation protocol

`allocate_buffer` and `allocate_image` are typed static customizations. An allocator exposes associated `buffer_type` and `image_type` types and returns allocation senders. The `buffer_allocator`, `image_allocator`, and `resource_allocator` concepts describe this contract. Resource values expose Vulkan handles to execution while retaining allocator-specific ownership tokens.

## Optional VMA implementation

`vkexec_vma` implements the allocation protocol with VMA. A user allocator may model the same protocol. VMA is an optional module, not the memory model of core execution.

## Lifetime rules

The operation state and its context must remain valid until the operation completes. Vulkan objects referenced by submitted commands must remain valid until GPU completion. State referenced by `after_gpu()` must remain valid until that callback has run. Context-dependent owned objects must be destroyed before their context. Adopted instance/device lifetime remains the embedder's responsibility.
