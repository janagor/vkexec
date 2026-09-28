# Descriptor backends

A descriptor schema states shader binding semantics; a resource table holds the corresponding runtime Vulkan handles. Neither type stores descriptor heap offsets or mapped bytes. The descriptor strategy chooses how those logical bindings reach Vulkan commands.

## Descriptor sets

Core provides the descriptor-set strategy. Create or obtain a compute pipeline, bind storage resources to the layout, and use the resulting descriptor set with `compute_pass`. The [first program](../getting-started/first-program.md) walks through this path from pipeline creation to cleanup.

## Descriptor heap extension

Link `vkexec::ext_descriptor_heap` and negotiate its Vulkan requirements before creating the context. The compiled [descriptor heap example](https://github.com/janagor/vkexec/blob/main/examples/extensions/descriptor_heap/descriptor_heap.cpp) configures the extension and checks availability:

```{literalinclude} ../../examples/extensions/descriptor_heap/descriptor_heap.cpp
:language: cpp
:start-at: auto make_requirements()
:end-at: return requirements;
```

The example then queries heap layout, allocates a device-addressable storage buffer and descriptor heap buffer, writes a storage descriptor, creates a pipeline with `vkexec::descriptor_heap`, and dispatches a pass using the same strategy. Heap indices, sizes, and procedure pointers stay inside the extension API and its lowering path.

```{literalinclude} ../../examples/extensions/descriptor_heap/descriptor_heap.cpp
:language: cpp
:start-at: auto pipe = vkexec::examples::sync_wait_value(vkexec::factory::make_compute_pipeline(vkexec::descriptor_heap,
:end-at: descriptor_heap.comp
```

Do not substitute descriptor-set state for heap state or assume every device supports the extension. See [the extension architecture](../architecture/extensions.md) and [API reference](../api/extensions.md).
