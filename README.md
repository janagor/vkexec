# vkexec

[![ci](https://github.com/janagor/vkexec/actions/workflows/ci.yml/badge.svg)](https://github.com/janagor/vkexec/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/janagor/vkexec/branch/main/graph/badge.svg)](https://codecov.io/gh/janagor/vkexec)
[![CodeQL](https://github.com/janagor/vkexec/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/janagor/vkexec/actions/workflows/codeql-analysis.yml)

## About

`vkexec` is a C++23 library that provides a **stdexec Vulkan compute backend** with a JIT tracing eDSL:

1. Trace C++ operators on `vkexec::Float` / `vkexec::Int` into an AST
2. Emit GLSL, compile to SPIR-V via glslang, and cache `VkPipeline`s
3. Dispatch with `vkexec::bulk`, or sequence several `compute_pass`es plus barriers in one submit

```cpp
#include <vkexec/vkexec.hpp>
#include <stdexec/execution.hpp>

namespace ex = stdexec;

struct SimParams { float dt; float damping; };
BOOST_DESCRIBE_STRUCT(SimParams, (), (dt, damping))

int main() {
  vkexec::context ctx;
  vkexec::buffer<float> positions(ctx, 10000, 0.0f);
  vkexec::buffer<float> velocities(ctx, 10000, 1.5f);
  SimParams params{ 0.016f, 0.99f };

  auto pipeline = ex::schedule(ctx.get_scheduler())
    | vkexec::bulk(10000, params, [&](vkexec::Int idx, vkexec::PushConstant<SimParams> pc) {
        vkexec::Float p = positions[idx];
        vkexec::Float v = velocities[idx];
        v = v * pc.get<&SimParams::damping>();
        p = p + (v * pc.get<&SimParams::dt>());
        positions[idx] = p;
        velocities[idx] = v;
      });

  ex::sync_wait(pipeline);

  // Several compute kernels in one command buffer:
  auto graph = ex::schedule(ctx.get_scheduler())
    | vkexec::compute_pass(10000, params, /* kernel */)
    | vkexec::barrier::compute_to_compute()
    | vkexec::compute_pass(10000, params, /* kernel */);
  ex::sync_wait(std::move(graph));
}
```

### Existing SPIR-V (hybrid)

Keep hand-written shaders. vkexec caches the pipeline and records dispatch. Push constants are a host POD (`upload_push_constants` / `compute_pass`) — not `push_constant<T>::get<&...>()`:

```cpp
struct ProjectPush { float view[16]; float projection[16]; std::uint64_t gaussian_addr; std::uint32_t splat_count; };

auto pipe = vkexec::compute_pipeline::from_glsl(ctx, glsl, vkexec::layout_desc{
  .bindings = { vkexec::buffer_access::readonly, vkexec::buffer_access::writeonly },
  .push_constant_size = sizeof(ProjectPush),
  .specialization = { splat_count },
  .local_size = { 64, 1, 1 },
});
// or from_spirv(ctx, spirv, layout) when you already have .spv
VkDescriptorSet set = pipe.allocate_set();
pipe.update_set(set, buffers);
ex::sync_wait(ex::schedule(ctx.get_scheduler()) | vkexec::compute_pass(pipe, set, push, splat_count));
// or, with your own command buffer:
vkexec::upload_push_constants(cmd, pipe, push);
```

Build and run the sample:

```bash
nix develop
cmake --preset unixlike-clang-release
cmake --build out/build/unixlike-clang-release -j12
./out/build/unixlike-clang-release/src/vkexec_example/vkexec_example
./out/build/unixlike-clang-release/src/vkexec_sort_example/vkexec_sort_example
./out/build/unixlike-clang-release/src/vkexec_examples/passes
./out/build/unixlike-clang-release/src/vkexec_examples/spirv
./out/build/unixlike-clang-release/src/vkexec_examples/triangle
```

### Triangle window

Requires `vkexec_graphics` (GLFW + swapchain). Shaders are traced from C++. Each frame is a stdexec pipeline (same shape as compute `bulk`):

```cpp
#include <vkexec_graphics/vkexec_graphics.hpp>

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
