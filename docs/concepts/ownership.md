# Ownership

Public resource types follow distinct lifetime roles. `handles::` groups borrowed Vulkan handles and associated metadata. The caller owns the objects and uses the relevant explicit `destroy(ctx, handles)` operation where vkexec created a handle bundle. `owned::` provides move-only RAII wrappers. The optional `vma::` module owns VMA-backed buffers, images, and allocators.

Pass graph steps, bindings, and schedulers borrow what they name. A borrowed pipeline or buffer cannot be destroyed just after submission: keep it alive through GPU completion and any `after_gpu` action. Context-dependent owned objects must be destroyed before their context. An adopted instance or device remains the embedder's responsibility and must outlive the vkexec context.

`buffer<T, B>` and `tensor<T, B>` retain an owning buffer resource `B`; they are independent of a particular allocator. Context adoption and allocator adoption are separate operations. See [Adopting a context](../guides/adopting-context.md), [VMA](../guides/vma.md), and [the ownership architecture](../architecture/ownership.md).
