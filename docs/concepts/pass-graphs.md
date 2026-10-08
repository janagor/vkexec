# Pass graphs

A pass graph is an ordered collection of GPU pass steps. Static composition keeps the concrete step types in `pass_graph_sender<Steps...>`; this lets several adjacent operations lower into one graph without an accidental submission boundary. A dynamic pass graph is available when the graph structure is genuinely chosen at runtime.

`connect()` transfers the graph into an operation state with a receiver. During `start()`, the steps record into a command buffer in order. A barrier adaptor between passes describes a GPU memory and execution dependency, so a compute → barrier → compute sequence can remain one submission.

The compiled [multi-pass program](../examples/multi-pass.md) shows this sequence. For how semantic operations become primitive steps, see [Domain lowering](../architecture/domain-lowering.md).

## Runtime branches

Use `dynamic_pass_graph_sender::append_after()` when passes form a DAG. Nodes are added in topological order. The method returns a node index; pass the indices of required predecessors to later nodes. An empty predecessor list starts an independent branch. A node with multiple predecessors joins branches. `append()` and pipe composition retain their usual sequential dependency when used after DAG nodes.

Set queue affinity with `append_queue()` before adding each node. Resource declarations on passes let the planner add missing dependencies for conflicting uses, including queue ownership transfers. The executor submits batches without a host wait. When several branches remain terminal, an internal join submission waits for all of them before the sender completes. With presentation, the final graphics batch serves as that join.

Resource state tracking currently remembers one last use per resource. It therefore orders even compatible read-only uses on different queues. This conservative edge keeps a later write dependent on both readers through transitive ordering; parallel read frontiers require a separate state-tracking change.
