# First compute program

This walkthrough follows the compiled [`compute_execution.cpp`](https://github.com/janagor/vkexec/blob/main/examples/compute_execution.cpp) program. Build it with `-Dvkexec_BUILD_EXAMPLES=ON`, then run the `compute_execution` executable in your build tree. It uses GLSL, two host-visible buffers, one compute dispatch, and a result check.

## Create a context

`factory::make_context` creates vkexec's Vulkan execution context. The example enables validation layers. Its example helper waits for the factory sender and extracts the context; applications can use the public `vkexec::sync_wait` or `vkexec::try_sync_wait_value` boundaries.

```{literalinclude} ../../examples/compute_execution.cpp
:language: cpp
:start-at: auto ctx = vkexec::examples::sync_wait_value
:end-at: auto allocator = vkexec::examples::make_vma_allocator(*ctx);
```

The context owns or coordinates the device, queues, command pool, and completion machinery. Its scheduler carries the vkexec execution domain into later sender composition.

## Allocate resources

The optional VMA module supplies an allocator; two owning buffers hold positions and velocities. The initial values are `0.0F` and `k_initial_velocity`. Their Vulkan handles will later be borrowed by the binding API.

```{literalinclude} ../../examples/compute_execution.cpp
:language: cpp
:start-at: auto positions = vkexec::examples::sync_wait_value
:end-at: k_initial_velocity));
```

Allocation is independent of context adoption. An application that already owns a device can adopt it and still choose VMA, a custom allocator, or its own raw buffers.

## Create the compute pipeline

The embedded GLSL shader updates velocity and position. `layout_desc` describes two read/write storage bindings, push constants, and a workgroup size. `create` returns a result; the example checks it before taking the pipeline resources.

```{literalinclude} ../../examples/compute_execution.cpp
:language: cpp
:start-at: using enum vkexec::buffer_access;
:end-at: auto resources = vkexec::expected_take(resources_result);
```

The source shader is in the same compiled file. In an application, the SPIR-V path can avoid runtime GLSL tooling.

## Bind the buffers

Each `storage_binding` borrows a buffer handle and supplies the size and shader binding slot. `bind_storage` creates descriptor state matching the pipeline layout.

```{literalinclude} ../../examples/compute_execution.cpp
:language: cpp
:start-at: std::array<vkexec::storage_binding, 2> const buffers{
:end-at: auto bound = vkexec::expected_take(bound_result);
```

Keep both buffers and descriptor state alive until GPU completion. Binding alone does not submit commands.

## Compose and execute the sender

The scheduled predecessor selects the context. `compute_pass` adds a semantic GPU operation. Constructing that pipeline still does no GPU work. The compiled example uses a helper to present errors consistently in exception and no-exception builds:

```{literalinclude} ../../examples/compute_execution.cpp
:language: cpp
:start-at: sim_params const params
:end-at: static_cast<std::uint32_t>(k_element_count)));
```

The public blocking boundary is `vkexec::sync_wait(sender)`; `vkexec::try_sync_wait(sender)` always returns an outcome containing values, error, or stopped state. A receiver can instead connect to the sender and call `start()` to initiate work without blocking. Successful GPU submission returns from `start()` asynchronously. Read [the execution model](../concepts/execution-model.md) before replacing the blocking boundary.

## Check the result

After waiting, the example checks the beginning, middle, and end of the output buffers against the expected position and damped velocity. The wait is what makes those host reads safe.

```{literalinclude} ../../examples/compute_execution.cpp
:language: cpp
:start-at: float const expected_v
:end-at: vkexec execution sim ok:
```

## Release resources

The example frees descriptor state and pipeline resources explicitly. Owning VMA buffers and the allocator then destruct before the context. For an adopted context, the embedder's Vulkan objects must outlive the vkexec context and every submitted operation.

```{literalinclude} ../../examples/compute_execution.cpp
:language: cpp
:start-at: vkexec::free_compute_set(*ctx, resources, bound.set);
:end-at: vkexec::destroy(*ctx, resources);
```

Continue with [multi-pass execution](../examples/multi-pass.md) and [resource ownership](../concepts/ownership.md).
