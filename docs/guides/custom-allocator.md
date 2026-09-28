# Custom allocators

Core execution uses Vulkan handles and has no VMA dependency. A custom allocator supplies an owning buffer and/or image type plus allocation senders; `vkexec::allocate_buffer` and `vkexec::allocate_image` dispatch through `tag_invoke`.

## Implement the resource types

A buffer resource is movable and exposes `handle() -> VkBuffer` and `size()`. An image resource is movable and exposes `handle() -> VkImage` and `format()`. Mapped buffer helpers additionally require `mapped()`; typed writes require `flush()`, and readback requires `invalidate()`. The exact concepts are defined in [`resource_allocator.hpp`](https://github.com/janagor/vkexec/blob/main/include/vkexec/resource_allocator.hpp).

The test suite compiles this small resource model:

```{literalinclude} ../../test/resource_allocator_tests.cpp
:language: cpp
:start-after: docs: custom allocator resource types begin
:end-before: docs: custom allocator resource types end
```

The test objects use null Vulkan handles, so they check the C++ protocol rather than actual GPU allocation.

## Return typed allocation senders

Declare `buffer_type` and `image_type` on the allocator. Implement ADL-visible `tag_invoke` overloads that return senders completing with those exact types. The result can be infallible or report `vkexec::error`; a different error type does not satisfy the allocator concept.

```{literalinclude} ../../test/resource_allocator_tests.cpp
:language: cpp
:start-after: docs: custom allocator customizations begin
:end-before: docs: custom allocator customizations end
```

An allocation sender should defer work until start, as in the test. A production allocator must create valid Vulkan objects, retain their allocation tokens, and release them after GPU use ends. Core recording only borrows the resource handles; it never queries allocator-specific state.

## Choose the right concept

`buffer_allocator<A>` checks buffer creation, `image_allocator<A>` checks images, and `resource_allocator<A>` requires both. `mapped_buffer_allocator` adds the mapped buffer capabilities used by typed helpers. The public `buffer<T, B>` and `tensor<T, B>` templates retain the resource type `B`, so a custom allocator can integrate without changing GPU execution. See [Resources](../concepts/resources.md) and [VMA](vma.md) for a concrete implementation.
