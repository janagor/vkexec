# Resources

Core execution records work using Vulkan handles such as `VkBuffer`, `VkImage`, pipeline, layout, and descriptor-set handles. Its pass and binding interfaces do not ask which allocator created the memory. Resource tables hold logical bindings with handles, sizes, and layouts; descriptor backend placement is chosen later.

The allocation protocol is typed. `allocate_buffer(allocator, info)` and `allocate_image(allocator, info)` return senders whose value types match the allocator's declared `buffer_type` and `image_type`. `buffer_allocator`, `image_allocator`, and `resource_allocator` concepts validate that contract. Typed buffer and tensor helpers retain those owning resources.

Lifetime and synchronization remain application responsibilities. Keep owners alive until GPU completion, flush mapped writes before device use, invalidate mapped readback where required, and place barriers between conflicting GPU accesses. Read [Custom allocators](../guides/custom-allocator.md) for the protocol and [Barriers](../guides/barriers.md) for pass-local ordering.
