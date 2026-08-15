# vkexec

[![ci](https://github.com/janagor/vkexec/actions/workflows/ci.yml/badge.svg)](https://github.com/janagor/vkexec/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/janagor/vkexec/branch/main/graph/badge.svg)](https://codecov.io/gh/janagor/vkexec)
[![CodeQL](https://github.com/janagor/vkexec/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/janagor/vkexec/actions/workflows/codeql-analysis.yml)

## About

`vkexec` is a C++23 library that provides a **stdexec Vulkan compute backend** with a JIT tracing eDSL:

1. Trace C++ operators on `vkexec::Float` / `vkexec::Int` into an AST
2. Emit GLSL, compile to SPIR-V via glslang, and cache `VkPipeline`s
3. Dispatch with `vkexec::bulk` on a Vulkan compute queue

```cpp
#include <vkexec/vkexec.hpp>
#include <stdexec/execution.hpp>

namespace ex = stdexec;

struct SimParams { float dt; float damping; };
VKEXEC_PUSH_CONSTANT(SimParams, float, dt, float, damping);

int main() {
  vkexec::context ctx;
  vkexec::buffer<float> positions(ctx, 10000, 0.0f);
  vkexec::buffer<float> velocities(ctx, 10000, 1.5f);
  SimParams params{ 0.016f, 0.99f };

  auto pipeline = ex::schedule(ctx.get_scheduler())
    | vkexec::bulk(10000, params, [&](vkexec::Int idx, vkexec::PushConstant<SimParams> pc) {
        vkexec::Float p = positions[idx];
        vkexec::Float v = velocities[idx];
        v = v * pc.damping;
        p = p + (v * pc.dt);
        positions[idx] = p;
        velocities[idx] = v;
      });

  ex::sync_wait(pipeline);
}
```

Build and run the sample:

```bash
nix develop
cmake --preset unixlike-clang-release
cmake --build out/build/unixlike-clang-release -j12
./out/build/unixlike-clang-release/src/vkexec_example/vkexec_example
./out/build/unixlike-clang-release/src/vkexec_sort_example/vkexec_sort_example
./out/build/unixlike-clang-release/src/vkexec_triangle_example/vkexec_triangle_example
```

### Triangle window

Shaders are traced from C++. Each frame is a stdexec pipeline (same shape as compute `bulk`):

```cpp
vkexec::window win({ .width = 800, .height = 600, .title = "triangle" });
vkexec::graphics_pipeline pipeline(win.ctx(), win.render_pass(),
  [](vkexec::Int vid, vkexec::VertexWriter out) { /* ... */ },
  [](vkexec::FragmentReader in, vkexec::FragmentWriter out) { /* ... */ });

while (!win.should_close()) {
  win.poll_events();
  ex::sync_wait(
    ex::schedule(win.ctx().get_scheduler())
    | vkexec::draw(win, pipeline, 3));
}
```


## More Details

 * [Dependency Setup](README_dependencies.md)
 * [Building Details](README_building.md)
 * [Docker](README_docker.md)
