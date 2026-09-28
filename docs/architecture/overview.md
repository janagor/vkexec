# Architecture overview

The diagrams below are generated from checked-in DOT source.

```{graphviz} diagrams/modules.dot
```

## Layer diagram

```text
stdexec sender pipeline
    -> vkexec semantic expressions
    -> vkexec::domain normalization and lowering
    -> typed primitive pass steps
    -> pass_graph_sender<Steps...>
    -> connect() / operation state
    -> start() / command recording and queue submission
    -> GPU / fence completion
    -> continues_on(context scheduler)
    -> after_gpu callbacks
    -> set_value / set_error / set_stopped
```

## Module boundaries

```text
vkexec                       core execution and allocator-neutral abstractions
  +-- vkexec_features        promoted/core Vulkan capability helpers
  +-- vkexec_graphics        graphics and presentation
  +-- vkexec_ext_descriptor_heap
  +-- vkexec_ext_dynamic_rendering
  +-- vkexec_ext_timeline_semaphore
  +-- vkexec_vma             optional allocation implementation
  `-- vkexec_tools           optional shader/tooling layer
```

Extension entry points and backend state belong in their extension; allocation policy belongs outside core execution.
