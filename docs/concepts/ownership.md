# Ownership

Public resource types follow distinct lifetime roles. `handles::` groups borrowed Vulkan handles and associated metadata. The caller owns the objects and uses the relevant explicit `destroy(ctx, handles)` operation where vkexec created a handle bundle. `owned::` provides move-only RAII wrappers. The optional `vma::` module owns VMA-backed buffers, images, and allocators.

Pass graph steps and bindings borrow what they name; schedulers retain the execution runtime. A borrowed pipeline or buffer cannot be destroyed just after submission: keep it alive through GPU completion and any `after_gpu` action. Context-dependent owned objects requiring explicit cleanup must be destroyed while the public context is available. An adopted instance or device remains the embedder's responsibility and must outlive every user of the retained runtime.

Release all vkexec execution objects before C++ static object destruction begins. The process-wide runtime retirement agent shuts down during that phase, so schedulers, senders, and operation states should not be static or global objects.

`buffer<T, B>` and `tensor<T, B>` retain an owning buffer resource `B`; they are independent of a particular allocator. Context adoption and allocator adoption are separate operations. See [Adopting a context](../guides/adopting-context.md), [VMA](../guides/vma.md), and [the ownership architecture](../architecture/ownership.md).
