# Minimal compute

The compiled [`compute_execution.cpp`](https://github.com/janagor/vkexec/blob/main/examples/compute_execution.cpp) creates a Vulkan context, allocates position and velocity buffers, compiles a compute pipeline, binds descriptors, submits one pass, and checks the result. Follow the [first program tutorial](../getting-started/first-program.md) for each step and the lifetime rules.

The central sender composition comes directly from the example:

```{literalinclude} ../../examples/compute_execution.cpp
:language: cpp
:start-at: sim_params const params
:end-at: static_cast<std::uint32_t>(k_element_count)));
```

The example helper performs a synchronous wait and reports errors consistently across exception configurations. The public boundaries are `vkexec::sync_wait`, `vkexec::try_sync_wait`, or an asynchronously connected receiver.
