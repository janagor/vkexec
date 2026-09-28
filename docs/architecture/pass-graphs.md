# Pass graph representation

## Primitive pass steps

A static pass step is move-constructible and provides `record(context&, VkCommandBuffer, pass_cleanup&) -> status`. Recording happens while the command buffer is open. Steps may provide `after_gpu()` for work that must occur after GPU completion.

## Static pass graphs

`pass_graph_sender<Step0, Step1, ...>` stores a tuple of concrete step types. A statically composed sender chain retains those types and can append to an existing static graph without an extra submission boundary.

## Dynamic pass graphs

`dynamic_pass_graph_sender` stores a vector of type-erased `dynamic_pass_step` objects. It is the runtime construction escape hatch. Its `append()` lowers primitive semantic operations; statically known sender composition should remain a typed graph.

## Composite operations

A composite operation expands through `lower_vkexec_expression`, for example into binding, a barrier, and dispatch. It is normalized before step collection so its internal operations participate in the surrounding graph.

## Post-GPU actions

`after_gpu()` runs after GPU completion and transfer to the context scheduler, but before the downstream receiver's value completion. It is intended for work that must wait until GPU use of the pass resources has ended, including cleanup of temporary state. Exceptions from the callback are converted to `set_error` when exception support is enabled.
