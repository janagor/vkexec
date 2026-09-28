# Multi-pass compute

The compiled [`passes.cpp`](https://github.com/janagor/vkexec/blob/main/examples/passes.cpp) adds to a buffer, places a compute-to-compute barrier, then multiplies it. The two passes and barrier compose as one ordered graph. The final check expects `(initial + add) * scale`.

```{literalinclude} ../../examples/passes.cpp
:language: cpp
:start-at: auto graph = ex::schedule(ctx->get_scheduler())
:end-at: vkexec::examples::sync_wait_graph(graph);
```

`barrier::compute_to_compute()` orders the second dispatch after writes from the first. It does not create a host wait or a separate graph submission. See [Pass graphs](../concepts/pass-graphs.md) and [Barriers](../guides/barriers.md).
