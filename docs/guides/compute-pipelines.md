# Compute pipelines

Create a pipeline resource with the desired SPIR-V or GLSL path, layout description, bindings, push constants, and local size. Bind storage resources, then pipe `compute_pass` onto a sender scheduled on the context. The [compiled compute execution program](../examples/minimal-compute.md) shows the full sequence.
