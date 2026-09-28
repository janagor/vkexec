# Design principles

## stdexec is the execution vocabulary

Vulkan work composes as stdexec senders and sender adaptors. vkexec adds a Vulkan interpretation of that vocabulary, not a separate task system.

## Describe work first, execute later

Building a pipeline describes work. It does not record commands or submit to a queue. The sender/receiver protocol controls execution.

## Typed composition by default

Semantic expressions and statically known pass graphs preserve concrete types through lowering. Runtime graph construction may use explicit type erasure.

## GPU execution is asynchronous

On successful submission, a pass graph's `start()` returns without waiting for the GPU. Blocking belongs at an explicit synchronous consumer such as `sync_wait`.

## Backend mechanisms stay behind semantic APIs

Public algorithm names describe work. Descriptor and Vulkan capability mechanisms are selected during lowering or in their own modules.

## Ownership is explicit

Execution uses borrowed Vulkan handles. Owning wrappers and allocation protocols are separate layers. A context may own or adopt its instance and device.

Core is not a replacement for stdexec, a universal Vulkan allocator, or a facade that hides Vulkan handles. It does not emulate every descriptor backend through another. Dynamic graphs are available when runtime construction requires them, but typed graphs are preferred for statically known pipelines.
