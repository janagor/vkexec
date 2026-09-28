# Descriptor and extension architecture

## Descriptor schema

`descriptor_schema<...>` is the compile-time shader/resource contract. It validates logical binding slots and can derive classic descriptor-set layouts. It expresses semantics, not physical heap placement.

## Resource table

`resource_table` holds logical runtime buffer, image-view, and sampler bindings with Vulkan handles, sizes, and layouts. It has no descriptor-heap indices, strides, or mapped heap bytes.

## Descriptor strategy

The strategy selects backend lowering. Core exposes `descriptor_sets`; the descriptor-heap extension supplies `descriptor_heap`. Public operations such as `bind_resources` remain pipeable pass operations.

## Descriptor-set lowering

The descriptor-set backend maps schema/table semantics to set layouts, writes, binding commands, and bound values. Schema-derived layouts preserve explicit slots; handwritten layout descriptions may use positional bindings.

## Descriptor-heap extension

Heap resource and sampler indices, descriptor sizes and strides, mapped bytes, and heap-specific creation metadata remain in the extension's lowering environment and implementation. Backend-specific commands and procedure loading stay there. The caller or backend supplies physical heap indices.

## What core intentionally does not provide

Core provides neither descriptor-set-to-heap emulation nor a global heap slot allocator. Schema and resource tables remain backend-neutral.



## Core requirements

Context creation merges library baseline requirements with caller requirements. Callers cannot remove required floors. Adopted contexts report the capabilities enabled by their embedder.

## Promoted Vulkan features

`vkexec_features` provides helpers for capabilities available through core Vulkan versions or promoted extension paths. Negotiated API version and enabled features determine the available implementation path.

## Optional extensions

Descriptor heap, dynamic rendering, and timeline semaphore support live in their own extension modules. Core semantic types should not acquire extension-specific fields merely to serve one backend.

## Backend-specific procedure loading

Core execution loads the procedures needed by core capabilities. Where a capability was promoted from an extension, core may resolve either the core or compatible KHR entry point. Entry points that belong exclusively to an optional vkexec extension are resolved and stored by that extension module.



## Adding a new pass algorithm

Define a semantic algorithm/CPO that returns a `stdexec::sender_adaptor_closure` expression with a tag and data. For a primitive operation, define a step satisfying the pass-step recording contract and a `lower_vkexec_pass_step(Tag, Data, Env)` customization. Keep recording mechanics in the step.

## Adding a composite semantic operation

Define `lower_vkexec_expression(...)` to expand the operation into other semantic expressions. Expansion must make structural progress and must not introduce an intermediate submission boundary. Let normal pass collection gather the resulting primitives.

## Adding a backend lowering

Add backend-specific lowering or customization at the strategy boundary. Keep backend indices, procedure pointers, command details, and bound values out of backend-neutral schema and resource-table types.

## Adding an optional Vulkan extension

Put extension-specific public APIs, capability negotiation, procedure loading, and compiled implementation in the extension module. Share core concepts where they already express the same semantics.

## When type erasure is appropriate

Use `dynamic_pass_graph_sender` when the graph's structure genuinely depends on runtime construction. A statically composed pipeline should keep `pass_graph_sender<Steps...>` and its concrete step types.
