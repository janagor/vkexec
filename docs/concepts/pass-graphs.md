# Pass graphs

A pass graph is a collection of GPU pass steps with explicit dependencies. Static composition keeps the concrete step types in `pass_graph_sender<Steps...>`; this lets several adjacent operations lower into one graph without an accidental submission boundary. A dynamic pass graph is available when the graph structure is genuinely chosen at runtime.

`connect()` transfers the graph into an operation state with a receiver. During `start()`, the graph plans its batches and records steps into graph-owned command buffers according to their dependencies. A barrier adaptor between passes describes a GPU memory and execution dependency, so a compute → barrier → compute sequence can remain one submission.

The compiled [multi-pass program](../examples/multi-pass.md) shows this sequence. For how semantic operations become primitive steps, see [Domain lowering](../architecture/domain-lowering.md).

## Typed branches

`vkexec::when_all()` composes a GPU graph and is distinct from `stdexec::when_all`, which combines sender completions and values. It accepts two or more pass adaptor closures, and each branch must contain at least one pass. Each branch starts after the same incoming frontier. Its final pass becomes a frontier node; the next ordinary pass depends on every branch's frontier. The adaptor adds no GPU join submission. Nested branches work the same way, and `on_queue()` inside a branch applies only to that branch. A pass after `when_all()` keeps the queue selected before the branches.

```cpp
auto graph = stdexec::schedule(ctx.get_scheduler())
             | root_pass()
             | vkexec::when_all(
                 vkexec::on_queue(compute) | compute_pass_a() | compute_pass_b(),
                 vkexec::on_queue(graphics) | graphics_pass())
             | composite_pass();
```

Here `composite_pass()` depends on both `compute_pass_b()` and `graphics_pass()`. If the graph ends at `when_all()`, the existing terminal join waits for both branches. Presentation remains an outer graph boundary; place `present()` after the branches and any final graphics pass.

## Runtime branches

Use `dynamic_pass_graph_sender::append_after()` when passes form a DAG. Nodes are added in topological order. The method returns a node index; pass the indices of required predecessors to later nodes. An empty predecessor list starts an independent branch. A node with multiple predecessors joins branches. `append()` and pipe composition retain their usual sequential dependency when used after DAG nodes.

Set queue affinity with `append_queue()` before adding each node. Resource declarations on passes let the planner add missing dependencies for conflicting uses, including queue ownership transfers. The executor submits batches without a host wait. When several branches remain terminal, an internal join submission waits for all of them before the sender completes. With presentation, the final graphics batch serves as that join.

Resource state tracking currently remembers one last use per resource. It therefore orders even compatible read-only uses on different queues. This conservative edge keeps a later write dependent on both readers through transitive ordering; parallel read frontiers require a separate state-tracking change.
