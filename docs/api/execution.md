# Execution

Semantic pass adaptors, pass graphs, barriers, and submission.

## Public headers

- [`<vkexec/execution.hpp>`](https://github.com/janagor/vkexec/blob/main/include/vkexec/execution.hpp)
- [`<vkexec/pass.hpp>`](https://github.com/janagor/vkexec/blob/main/include/vkexec/pass.hpp)
- [`<vkexec/barrier.hpp>`](https://github.com/janagor/vkexec/blob/main/include/vkexec/barrier.hpp)

Caller-owned primary command buffers can use [`record_into()`](../record_into.md).
Its move-only `recorded_graph` retains temporary resources until explicit `complete()` or `abandon()`.

## Representative declarations

```{doxygenstruct} vkexec::handles::compute_pipeline
:project: vkexec
:outline:
```
