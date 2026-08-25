# vkexec

[![ci](https://github.com/janagor/vkexec/actions/workflows/ci.yml/badge.svg)](https://github.com/janagor/vkexec/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/janagor/vkexec/branch/main/graph/badge.svg)](https://codecov.io/gh/janagor/vkexec)
[![CodeQL](https://github.com/janagor/vkexec/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/janagor/vkexec/actions/workflows/codeql-analysis.yml)

## About

`vkexec` is a C++23 library that provides a **stdexec Vulkan compute backend** with a tracing eDSL, plus bindless (descriptor-heap) and optional GLFW graphics helpers suitable for embedding:

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
  auto [positions, velocities] = ex::sync_wait(ex::when_all(
                                   vkexec::buffer<float>::allocate(ctx, 10000, 0.0f),
                                   vkexec::buffer<float>::allocate(ctx, 10000, 1.5f)))
                                 .value();
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

### Device requirements and adopt

`vulkan_requirements` is caller-driven: you declare API floors, extensions, and `VkPhysicalDevice*Features` structs; vkexec merges them with a thin library baseline and selects a matching device. Prefer `context::try_create` when probing optional capabilities (`errc::unsupported` on mismatch); the throwing `context` constructor remains for hard failures.

Embedders that already own a Vulkan device (for example a Filament-like driver) can wrap it without transferring ownership:

```cpp
auto ctx = vkexec::context::adopt({
  .instance = instance,
  .physical_device = phys,
  .device = device,
  .allocator = vma,              // or null → vkexec creates one
  .compute_queue = compute_q,
  .compute_queue_family = compute_family,
  .graphics_queue = graphics_q,  // optional; defaults to compute
  .present_queue = present_q,    // optional; defaults to graphics
});
```

When extensions such as `VK_EXT_descriptor_heap` / `VK_EXT_shader_object` push-data are enabled, `context::procs()` caches the device PFNs (null when unavailable).

### Bindless (descriptor heap)

Hybrid apps can skip classic descriptor sets. Create a null-layout pipeline with `layout_desc.descriptor_heap = true`, allocate a `gpu_buffer` with `gpu_buffer_memory::descriptor_heap` (optionally `shader_device_address`), write storage descriptors into host-mapped heap memory, then bind + push data on the command buffer:

```cpp
auto layout = vkexec::query_descriptor_heap_layout(ctx);
auto heap = vkexec::gpu_buffer::create(ctx, {
  .size = vkexec::descriptor_heap_byte_size(layout, slot_count),
  .memory = vkexec::gpu_buffer_memory::descriptor_heap,
  .shader_device_address = true,
});
vkexec::write_storage_buffer_descriptor(ctx, buffer_addr, buffer_size, heap.mapped().subspan(...));

auto pipe = vkexec::compute_pipeline::from_glsl(ctx, glsl, vkexec::layout_desc{
  .descriptor_heap = true,
  .push_constant_size = sizeof(Push),
  .local_size = { 64, 1, 1 },
});

// In a recorded command buffer (or via bindless compute_pass):
vkexec::cmd_bind_resource_heap(ctx, cmd, heap.device_address(), heap.size(),
  /* reserved_offset */, layout.min_resource_heap_reserved_range);
vkexec::cmd_push_data(ctx, cmd, push);
```

Supporting RAII: `gpu_buffer`, `image` / `image_view` / `sampler`, `timeline_semaphore`, `frame_ring` (WSI slot/image gating), plus `vkexec_graphics::swapchain` for borrowed surfaces. Dynamic rendering helpers live in `rendering.hpp`.

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
./out/build/unixlike-clang-release/src/vkexec_examples/heap_present
```

`heap_present` is a headless smoke of public Phase 2–4 APIs: optional descriptor-heap compute, dynamic rendering to an offscreen color target, then a few swapchain present frames.

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
