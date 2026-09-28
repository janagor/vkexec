# Pass graphs

A pass graph is an ordered collection of GPU pass steps. Static composition keeps the concrete step types in `pass_graph_sender<Steps...>`; this lets several adjacent operations lower into one graph without an accidental submission boundary. A dynamic pass graph is available when the graph structure is genuinely chosen at runtime.

`connect()` transfers the graph into an operation state with a receiver. During `start()`, the steps record into a command buffer in order. A barrier adaptor between passes describes a GPU memory and execution dependency, so a compute → barrier → compute sequence can remain one submission.

The compiled [multi-pass program](../examples/multi-pass.md) shows this sequence. For how semantic operations become primitive steps, see [Domain lowering](../architecture/domain-lowering.md).
