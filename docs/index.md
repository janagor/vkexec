# vkexec documentation

vkexec composes Vulkan work with stdexec senders. A scheduled sender carries the context; semantic pass adaptors describe GPU work; the consumer chooses when to wait.

```cpp
auto graph = ex::schedule(ctx->get_scheduler())
  | vkexec::compute_pass(resources, bound.set, params, work_count);
auto completion = vkexec::sync_wait(std::move(graph));
```

The [compiled compute program](https://github.com/janagor/vkexec/blob/main/examples/compute_execution.cpp) supplies the context, pipeline, bindings, and result check behind this expression. `sync_wait` is the explicit host boundary; a connected receiver can instead observe completion asynchronously.

vkexec provides:

- **stdexec-native composition:** senders and receivers carry success, error, and stopped completions.
- **Typed pass graphs:** statically composed passes retain concrete step types and ordered GPU barriers.
- **Asynchronous GPU execution:** successful `start()` submits without waiting for the GPU fence.

Start with [Getting Started](getting-started/index.md) for a compiled tutorial. Use the [API Reference](api/index.md) for public declarations and [Architecture](architecture/index.md) for implementation design. The [Concepts](concepts/index.md), [Guides](guides/index.md), [Examples](examples/index.md), and [Development](development/index.md) sections cover the next steps.

```{toctree}
:maxdepth: 2

getting-started/index
concepts/index
guides/index
examples/index
api/index
architecture/index
development/index
```
