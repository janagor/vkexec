# vkexec

[![CI](https://github.com/janagor/vkexec/actions/workflows/ci.yml/badge.svg)](https://github.com/janagor/vkexec/actions/workflows/ci.yml)
-[![codecov](https://codecov.io/gh/janagor/vkexec/branch/main/graph/badge.svg)](https://codecov.io/gh/janagor/vkexec)
-[![CodeQL](https://github.com/janagor/vkexec/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/janagor/vkexec/actions/workflows/codeql-analysis.yml)
[![Documentation](https://github.com/janagor/vkexec/actions/workflows/docs.yml/badge.svg)](https://janagor.github.io/vkexec/)

vkexec is a C++20 stdexec backend for composing asynchronous Vulkan compute and graphics work. It is an evolving project; review the public headers and examples when adopting an API.

## Example

A sender scheduled on a vkexec context can compose a compute pass:

```cpp
auto graph = ex::schedule(ctx->get_scheduler())
  | vkexec::compute_pass(resources, bound.set, params, work_count);
auto completion = vkexec::sync_wait(std::move(graph));
```

The pipeline and bindings come from the [compiled compute example](examples/compute_execution.cpp). The public `sync_wait` call is an explicit host blocking boundary; its return type depends on the exception configuration. The [first program](docs/getting-started/first-program.md) walks through the setup and result check.

## Execution model

- Construction describes GPU work.
- `start()` initiates submitted work asynchronously.
- `sync_wait()` is an explicit blocking boundary.

## What vkexec provides

- stdexec scheduler, semantic sender adaptors, and typed pass graphs.
- Compute passes, resource binding, and GPU barriers.
- Allocator-neutral resource interfaces with optional VMA-backed ownership.
- Graphics and presentation helpers plus optional Vulkan extensions.

## Modules

| CMake target | Purpose |
|---|---|
| `vkexec::vkexec` | Core execution and allocator-neutral resources |
| `vkexec::vma` | Optional VMA allocation |
| `vkexec::features` | Vulkan capability helpers |
| `vkexec::vkexec_graphics` | Graphics and presentation |
| `vkexec::tools` | Optional shader tooling |
| `vkexec::ext_descriptor_heap` | Descriptor heap backend |
| `vkexec::ext_dynamic_rendering` | Dynamic rendering |
| `vkexec::ext_timeline_semaphore` | Timeline semaphore helpers |

The [module architecture](docs/architecture/modules.md) explains their boundaries; CMake target definitions are authoritative for build options and dependencies.

## Requirements

CMake 3.29+, C++20, and a Vulkan loader are required. CI checks GCC 12, Clang 16, and MSVC 19.43 floor lanes. GPU examples need a Vulkan-capable device or software driver. The [build guide](docs/development/building.md) and [compiler support](docs/development/compiler-support.md) give details.

## Building

```sh
cmake -S . -B build -G Ninja -DBUILD_TESTING=ON -Dvkexec_BUILD_EXAMPLES=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Documentation

The [documentation site](https://janagor.github.io/vkexec/) is generated from [`docs/`](docs/index.md). Start with [Getting started](docs/getting-started/index.md), then read [Concepts](docs/concepts/index.md), [Guides](docs/guides/index.md), [API reference](docs/api/index.md), [Architecture](docs/architecture/index.md), or [Development](docs/development/index.md).

## Examples

Compiled programs live in [`examples/`](examples/). The [examples guide](docs/examples/index.md) identifies the program for each workflow.

## License

[Apache License 2.0](LICENSE).
