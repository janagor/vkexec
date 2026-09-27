# vkexec Architecture

## 1. Purpose

`README_design.md` describes the architecture and design invariants of vkexec. It is for contributors implementing execution algorithms, Vulkan backends, resources, and extensions. User-facing setup and examples belong in `README.md`; build configuration belongs in the build documentation.

## 2. Design principles

### 2.1 stdexec is the execution vocabulary

Vulkan work composes as stdexec senders and sender adaptors. vkexec adds a Vulkan interpretation of that vocabulary, not a separate task system.

### 2.2 Describe work first, execute later

Building a pipeline describes work. It does not record commands or submit to a queue. The sender/receiver protocol controls execution.

### 2.3 Typed composition by default

Semantic expressions and statically known pass graphs preserve concrete types through lowering. Runtime graph construction may use explicit type erasure.

### 2.4 GPU execution is asynchronous

On successful submission, a pass graph's `start()` returns without waiting for the GPU. Blocking belongs at an explicit synchronous consumer such as `sync_wait`.

### 2.5 Backend mechanisms stay behind semantic APIs

Public algorithm names describe work. Descriptor and Vulkan capability mechanisms are selected during lowering or in their own modules.

### 2.6 Ownership is explicit

Execution uses borrowed Vulkan handles. Owning wrappers and allocation protocols are separate layers. A context may own or adopt its instance and device.

Core is not a replacement for stdexec, a universal Vulkan allocator, or a facade that hides Vulkan handles. It does not emulate every descriptor backend through another. Dynamic graphs are available when runtime construction requires them, but typed graphs are preferred for statically known pipelines.

## 3. Architecture overview

### 3.1 Layer diagram

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

### 3.2 Module boundaries

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

## 4. Execution model

### 4.1 `context`

`context` coordinates the Vulkan device and queues, command pool, submission synchronization, host agent, and completion agent. It owns its internal machinery. `factory::make_context()` creates an owned instance/device; `factory::adopt_context()` borrows an embedder's handles and queues. A context is an execution resource, not itself a stdexec scheduler.

### 4.2 `scheduler`

`vkexec::scheduler` is associated with a context. `ex::schedule(ctx->get_scheduler())` produces a sender whose environment advertises both the vkexec domain and the context scheduler for value completion. Its value completion runs on the context's host agent. `schedule()` does not execute GPU commands; the sender environment carries the context information used for lowering subsequent GPU operations.

### 4.3 Scheduler environments

`scheduler_env` advertises the vkexec domain and the context scheduler for `set_value`. This is a value-completion guarantee; error and stopped completions need not run on that scheduler. A domain-only environment can advertise vkexec lowering without claiming a completion scheduler. Pass adaptors require a predecessor whose value completion is guaranteed on `vkexec::scheduler`.

### 4.4 `vkexec::domain`

`domain::transform_sender()` calls `lower_vkexec_sender` when vkexec has a lowering for the operation. Otherwise it delegates to `stdexec::default_domain`, so ordinary stdexec algorithms remain composable.

### 4.5 Semantic sender expressions

An algorithm such as `compute_pass(...)` returns a stdexec sender adaptor closure. Piping it onto a child sender creates an expression shaped as `sender_expr<Tag, Data, Child>`. Environment queries forward to the child until the domain transforms it. Construction does not record or submit Vulkan work.

Public composable algorithms should produce semantic expressions when the vkexec domain can lower them. Use `stdexec::sender_adaptor_closure` rather than a separate `operator|` mechanism.

### 4.6 Domain lowering

```text
algorithm(args...) -> expr_closure<Tag, Data>
  -> sender_expr<Tag, Data, Child>
  -> domain::transform_sender()
  -> normalize_vkexec_expression()
  -> lower_vkexec_pass_step()
  -> typed Step -> pass_graph_sender<Steps...>
```

`lower_vkexec_expression` expands a composite operation into other semantic expressions. Each expansion must make structural progress. Normalization completes before primitive steps are collected, so a composite does not accidentally create a submission boundary. `lower_vkexec_pass_step` turns a primitive operation into one recordable step. A `schedule_sender` predecessor materializes directly into a pass graph without an extra `let_value`; other scheduler-completing predecessors use `let_value` to preserve their completion dependency.

### 4.7 Pass graphs

A pass graph owns its ordered steps. `connect()` creates an operation state containing those steps and the receiver. It still does not submit work.

### 4.8 Submission and completion

On `start()`, the operation checks for a requested stop, opens and records a command buffer, then starts the fence-backed submission sender. The successful path submits and returns without waiting for GPU completion. After the fence signals, `ex::continues_on(fence_sender, ctx->get_scheduler())` transfers completion to the context host scheduler. `after_gpu` callbacks run there before the downstream receiver completes. The completion agent observes GPU progress; it is not an arbitrary executor for downstream user code.

`start() != wait for GPU`. `sync_wait(...)` is an explicit blocking boundary.

### 4.9 Cancellation and completion channels

The receiver observes `set_value` on success, `set_error(vkexec::error)` on failure, and `set_stopped` on cancellation. A stop requested before submission can prevent work from being submitted. Submitted work still follows the fence completion path; cancellation is not an error.

## 5. Pass-graph representation

### 5.1 Primitive pass steps

A static pass step is move-constructible and provides `record(context&, VkCommandBuffer, pass_cleanup&) -> status`. Recording happens while the command buffer is open. Steps may provide `after_gpu()` for work that must occur after GPU completion.

### 5.2 Static pass graphs

`pass_graph_sender<Step0, Step1, ...>` stores a tuple of concrete step types. A statically composed sender chain retains those types and can append to an existing static graph without an extra submission boundary.

### 5.3 Dynamic pass graphs

`dynamic_pass_graph_sender` stores a vector of type-erased `dynamic_pass_step` objects. It is the runtime construction escape hatch. Its `append()` lowers primitive semantic operations; statically known sender composition should remain a typed graph.

### 5.4 Composite operations

A composite operation expands through `lower_vkexec_expression`, for example into binding, a barrier, and dispatch. It is normalized before step collection so its internal operations participate in the surrounding graph.

### 5.5 Post-GPU actions

`after_gpu()` runs after GPU completion and transfer to the context scheduler, but before the downstream receiver's value completion. It is intended for work that must wait until GPU use of the pass resources has ended, including cleanup of temporary state. Exceptions from the callback are converted to `set_error` when exception support is enabled.

## 6. Resource and ownership model

### 6.1 Borrowed execution resources

Core execution consumes Vulkan handles such as `VkBuffer`, `VkImage`, pipelines, layouts, and descriptor sets. Recording does not ask a concrete allocator for memory. Borrowed resources must remain valid through GPU completion and any `after_gpu` work that uses them.

### 6.2 `handles::`

`handles::` types group borrowed Vulkan handles and associated metadata for execution. They do not transfer ownership of the underlying Vulkan objects.

### 6.3 `owned::`

`owned::` types are RAII wrappers where vkexec provides ownership. Their destruction requirements are separate from the borrowed handle types passed to execution algorithms.

### 6.4 Allocation protocol

`allocate_buffer` and `allocate_image` are typed static customizations. An allocator exposes associated `buffer_type` and `image_type` types and returns allocation senders. The `buffer_allocator`, `image_allocator`, and `resource_allocator` concepts describe this contract. Resource values expose Vulkan handles to execution while retaining allocator-specific ownership tokens.

### 6.5 Optional VMA implementation

`vkexec_vma` implements the allocation protocol with VMA. A user allocator may model the same protocol. VMA is an optional module, not the memory model of core execution.

### 6.6 Lifetime rules

The operation state and its context must remain valid until the operation completes. Vulkan objects referenced by submitted commands must remain valid until GPU completion. State referenced by `after_gpu()` must remain valid until that callback has run. Context-dependent owned objects must be destroyed before their context. Adopted instance/device lifetime remains the embedder's responsibility.

## 7. Descriptor architecture

### 7.1 Descriptor schema

`descriptor_schema<...>` is the compile-time shader/resource contract. It validates logical binding slots and can derive classic descriptor-set layouts. It expresses semantics, not physical heap placement.

### 7.2 Resource table

`resource_table` holds logical runtime buffer, image-view, and sampler bindings with Vulkan handles, sizes, and layouts. It has no descriptor-heap indices, strides, or mapped heap bytes.

### 7.3 Descriptor strategy

The strategy selects backend lowering. Core exposes `descriptor_sets`; the descriptor-heap extension supplies `descriptor_heap`. Public operations such as `bind_resources` remain pipeable pass operations.

### 7.4 Descriptor-set lowering

The descriptor-set backend maps schema/table semantics to set layouts, writes, binding commands, and bound values. Schema-derived layouts preserve explicit slots; handwritten layout descriptions may use positional bindings.

### 7.5 Descriptor-heap extension

Heap resource and sampler indices, descriptor sizes and strides, mapped bytes, and heap-specific creation metadata remain in the extension's lowering environment and implementation. Backend-specific commands and procedure loading stay there. The caller or backend supplies physical heap indices.

### 7.6 What core intentionally does not provide

Core provides neither descriptor-set-to-heap emulation nor a global heap slot allocator. Schema and resource tables remain backend-neutral.

## 8. Vulkan capabilities and extensions

### 8.1 Core requirements

Context creation merges library baseline requirements with caller requirements. Callers cannot remove required floors. Adopted contexts report the capabilities enabled by their embedder.

### 8.2 Promoted Vulkan features

`vkexec_features` provides helpers for capabilities available through core Vulkan versions or promoted extension paths. Negotiated API version and enabled features determine the available implementation path.

### 8.3 Optional extensions

Descriptor heap, dynamic rendering, and timeline semaphore support live in their own extension modules. Core semantic types should not acquire extension-specific fields merely to serve one backend.

### 8.4 Backend-specific procedure loading

Core execution loads the procedures needed by core capabilities. Where a capability was promoted from an extension, core may resolve either the core or compatible KHR entry point. Entry points that belong exclusively to an optional vkexec extension are resolved and stored by that extension module.

## 9. Synchronization

### 9.1 Pass-local barriers

Barrier adaptors describe GPU memory and execution dependencies inside a pass graph. A compute/barrier/compute chain records the barrier between the two dispatches in one ordered submission.

### 9.2 synchronization2 versus legacy lowering

A semantic barrier uses synchronization2 Flags2 and `vkCmdPipelineBarrier2` when available. Otherwise it uses checked translation to the legacy barrier command. Unsupported legacy translations fail through the pass graph's error channel. Adopted contexts use feature information supplied by the embedder.

### 9.3 Queue submission

Recording and queue submission are separate from barrier semantics. The context serializes command-pool and queue access as required. A submitted graph uses a fence-backed completion sender.

### 9.4 GPU completion versus host scheduling

The fence establishes GPU completion. `continues_on(ctx->get_scheduler())` then establishes the host context for cleanup and downstream value completion. A pass-local barrier does not serve this host/GPU completion role.

## 10. Error and cancellation model

### 10.1 `vkexec::error`

Expected Vulkan and vkexec failures are normalized into the `vkexec::error` completion channel.

### 10.2 `set_value`

Successful operations complete through `set_value` only after their promised work and required post-GPU cleanup are done.

### 10.3 `set_error`

Validation, recording, submission, completion, or cleanup failures complete through `set_error`. A failure before submission need not wait for the GPU.

### 10.4 `set_stopped`

Cancellation completes through `set_stopped`; it is not an error value. Stop checks occur before work begins where the sender protocol permits them.

### 10.5 Exception containment

Operation `start()` paths remain `noexcept`. Potentially throwing internals are caught when exception support is enabled and translated to the error channel. Exceptions must not escape through a receiver completion path.

### 10.6 Synchronous boundaries

Blocking consumers such as `sync_wait` are explicit synchronous boundaries. They do not change the asynchronous contract of a successfully submitted GPU pass sender.

## 11. Public/private boundaries

### 11.1 Public headers

`include/vkexec/*.hpp` provides the supported core API. Optional modules have their own include trees and namespaces.

### 11.2 `include/vkexec/detail`

Installed detail headers support public template implementation but are not public API. Detail types must not become public signature commitments.

### 11.3 `src/**/detail`

Compiled implementation headers remain private and are not installed.

### 11.4 stdexec private API quarantine

vkexec uses the public `<stdexec/execution.hpp>` vocabulary. Direct `stdexec::__*`, `ex::__*`, `exec::__*`, `STDEXEC::__*`, and `stdexec/__detail/*` dependencies are forbidden outside `include/vkexec/detail/stdexec_compat.hpp`. That compatibility header must not expose private stdexec types in public signatures.

### 11.5 vk-bootstrap boundary

vk-bootstrap is an implementation mechanism for instance/device discovery and creation. Public APIs use Vulkan handles and vkexec types, not vk-bootstrap types. Applications with their own device setup use `factory::adopt_context()`.

### 11.6 Optional dependency boundaries

VMA stays in `vkexec_vma`; extension implementation state and entry points stay in their extension modules. Core allocation and descriptor semantics do not depend on either mechanism.

## 12. Extending vkexec

### 12.1 Adding a new pass algorithm

Define a semantic algorithm/CPO that returns a `stdexec::sender_adaptor_closure` expression with a tag and data. For a primitive operation, define a step satisfying the pass-step recording contract and a `lower_vkexec_pass_step(Tag, Data, Env)` customization. Keep recording mechanics in the step.

### 12.2 Adding a composite semantic operation

Define `lower_vkexec_expression(...)` to expand the operation into other semantic expressions. Expansion must make structural progress and must not introduce an intermediate submission boundary. Let normal pass collection gather the resulting primitives.

### 12.3 Adding a backend lowering

Add backend-specific lowering or customization at the strategy boundary. Keep backend indices, procedure pointers, command details, and bound values out of backend-neutral schema and resource-table types.

### 12.4 Adding an optional Vulkan extension

Put extension-specific public APIs, capability negotiation, procedure loading, and compiled implementation in the extension module. Share core concepts where they already express the same semantics.

### 12.5 When type erasure is appropriate

Use `dynamic_pass_graph_sender` when the graph's structure genuinely depends on runtime construction. A statically composed pipeline should keep `pass_graph_sender<Steps...>` and its concrete step types.

## 13. Architectural invariants

- Public execution APIs use stdexec vocabulary and semantic sender expressions.
- `vkexec::domain` lowers vkexec operations and delegates unrelated operations to `stdexec::default_domain`.
- Composite lowering makes structural progress and does not add accidental submission boundaries.
- Statically known graphs retain concrete step types; runtime type erasure is isolated to dynamic graphs.
- A successful GPU pass `start()` submits without waiting; blocking is an explicit consumer choice.
- GPU completion transfers to the context scheduler before `after_gpu` and downstream value completion.
- Execution uses Vulkan handles and does not depend on a concrete allocator; VMA is optional.
- Descriptor schema and resource tables do not contain backend-specific heap state.
- Cancellation uses `set_stopped`, failures use `set_error(vkexec::error)`.
- Context-dependent resources do not outlive their context.
- Private stdexec APIs, vk-bootstrap types, and extension-specific entry points remain behind their boundaries.

## 14. Architecture enforcement tests

CTest registers `CheckStdexecPrivateApi.cmake`, `CheckVmaBoundary.cmake`, and `CheckVkBootstrapBoundary.cmake`. These checks guard the stdexec-private, allocator, and device-creation dependency boundaries. They complement ordinary behavioral tests; a passing build alone does not establish those architecture rules.
