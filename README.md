# vkexec

[![ci](https://github.com/janagor/vkexec/actions/workflows/ci.yml/badge.svg)](https://github.com/janagor/vkexec/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/janagor/vkexec/branch/main/graph/badge.svg)](https://codecov.io/gh/janagor/vkexec)
[![CodeQL](https://github.com/janagor/vkexec/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/janagor/vkexec/actions/workflows/codeql-analysis.yml)

## About

`vkexec` is a C++23 library that provides a **stdexec Vulkan compute backend** with prebuilt GLSL/SPIR-V shaders, plus bindless (descriptor-heap) and optional GLFW graphics helpers suitable for embedding:

1. Author shaders as GLSL strings or embedded SPIR-V
2. Build and cache `VkPipeline`s via `compute_pipeline` / `graphics_pipeline`
3. Dispatch with `compute_pass` (and chain barriers / multiple passes in one submit)

```cpp
#include <vkexec/buffer.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/sync_wait_outcome.hpp>
#include <vkexec/sync_wait.hpp>

#include <stdexec/execution.hpp>

namespace ex = stdexec;

struct sim_params { float dt; float damping; };

constexpr std::string_view k_sim_glsl = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer Positions { float data[]; } positions;
layout(set = 0, binding = 1) buffer Velocities { float data[]; } velocities;
layout(push_constant) uniform Push { float dt; float damping; } pc;
void main() {
  uint i = gl_GlobalInvocationID.x;
  float v = velocities.data[i] * pc.damping;
  positions.data[i] = positions.data[i] + v * pc.dt;
  velocities.data[i] = v;
}
)";

int main() {
  try {
    auto ctx = vkexec::detail::take_sync_value(*vkexec::sync_wait(vkexec::context::create()));
    auto positions = vkexec::detail::take_sync_value(*vkexec::sync_wait(vkexec::buffer<float>::allocate(*ctx, 10000, 0.0f)));
    auto velocities = vkexec::detail::take_sync_value(*vkexec::sync_wait(vkexec::buffer<float>::allocate(*ctx, 10000, 1.5f)));

    using enum vkexec::buffer_access;
    auto pipe = vkexec::detail::take_sync_value(*vkexec::sync_wait(vkexec::compute_pipeline::create(*ctx,
      k_sim_glsl,
      vkexec::layout_desc{ .bindings = { readwrite, readwrite }, .push_constant_size = sizeof(sim_params) },
      "sim.comp")));

    auto bound = vkexec::detail::take_sync_value(*vkexec::sync_wait(
      vkexec::compute_pipeline::bind_storage_sender(pipe, { positions, velocities })));

    sim_params params{ 0.016f, 0.99f };
    if (auto waited = vkexec::sync_wait(ex::schedule(ctx->get_scheduler())
                                        | vkexec::compute_pass(bound.pipeline, bound.set, params, 10000));
        !waited.has_value()) {
      return 1;
    }

    // Several compute kernels in one command buffer:
    auto graph = ex::schedule(ctx->get_scheduler())
      | vkexec::compute_pass(bound.pipeline, bound.set, params, 10000)
      | vkexec::barrier::compute_to_compute()
      | vkexec::compute_pass(bound.pipeline, bound.set, params, 10000);
    if (auto waited = vkexec::sync_wait(std::move(graph)); !waited.has_value()) { return 1; }
  } catch (vkexec::error const& err) {
    std::cerr << std::format("{}\n", err.message());
    return 1;
  }
}
```

### Error model

Public APIs are **senders** (stdexec). Completions follow stdexec semantics:

- **`set_value(...)`** — success
- **`set_error(vkexec::error)`** — failure (same error type everywhere async)
- **`set_stopped()`** — cancellation (not an error)

There is no public `std::expected`, `result<T>`, `status`, `value_or_throw`, or `create_sync`. Factory functions such as `context::create`, `buffer::allocate`, and `compute_pipeline::create` return senders; compose them with `stdexec::let_value` or block at the sync boundary with `sync_wait`.

- **`vkexec::error`** carries a `boost::system::error_code` plus optional detail text. Use `error.message()` for a human-readable string.
- **`vkexec::errc`** covers library-level failures (`invalid_argument`, `unsupported`, `cancelled`, …).
- **Vulkan failures** use `vkexec::make_vk_error_code(VkResult)` / `vkexec::make_vk_error(...)`.

**Sync boundary (`VKEXEC_ENABLE_EXCEPTIONS`, default ON):** `vkexec::sync_wait(sender)` delegates to `stdexec::sync_wait` and **throws `vkexec::error`** on `set_error`. A disengaged `std::optional` means `set_stopped()` (not an error).

```cpp
try {
  auto ctx = vkexec::detail::take_sync_value(*vkexec::sync_wait(vkexec::context::create({ .requirements = reqs })));
  auto waited = vkexec::sync_wait(vkexec::buffer<float>::allocate(*ctx, n, fill));
  if (!waited.has_value()) { /* stopped */ }
  auto buf = vkexec::detail::take_sync_value(std::move(*waited));
} catch (vkexec::error const &err) {
  std::cout << std::format("{}\n", err.message());
}
```

**`-fno-exceptions` builds (`-DVKEXEC_ENABLE_EXCEPTIONS=OFF`):** `sync_wait` returns `vkexec::sync_wait_outcome<...>` with `values`, `error`, and `stopped` fields — no throwing, no `std::expected`.

```cpp
auto outcome = vkexec::sync_wait(vkexec::context::create({ .requirements = reqs }));
if (outcome.failed()) {
  std::cout << std::format("{}\n", outcome.take_error().message());
  return;
}
if (outcome.stopped || !outcome.values.has_value()) { /* stopped */ return; }
auto ctx = vkexec::detail::take_sync_value(std::move(*outcome.values));
```

For tests without exceptions, use `vkexec::try_sync_wait` (same `sync_wait_outcome` shape).

Embedders that do not use stdexec pipes can block on factory senders directly:

```cpp
auto ctx = vkexec::sync_wait_value(vkexec::context::adopt({ /* borrowed handles */ }));
auto pipe = vkexec::sync_wait_value(vkexec::compute_pipeline::create(*ctx, spirv, layout));

if (auto heap = vkexec::try_sync_wait_value(vkexec::gpu_buffer::create(*ctx, info)); heap) {
  // use *heap
}
```

`sync_wait_value` throws `vkexec::error` on failure or stop (when exceptions are enabled). `try_sync_wait_value` returns `vkexec::result<T>` instead.

Common entry points:

| API | Returns |
|-----|---------|
| `context::create` / `context::adopt` | sender → `set_value(std::unique_ptr<context>)` |
| `buffer<T>::allocate` / `create` | sender → `set_value(buffer<T>)` |
| `compute_pipeline::create` | sender → `set_value(compute_pipeline)` |
| `compute_pipeline::bind_storage_sender` | sender → `set_value(bound_compute_pipeline)` |
| `window::create` / `window::headless` | sender → `set_value(window)` |
| `graphics_pipeline::create` | sender → `set_value(graphics_pipeline)` |
| `mesh::create` | sender → `set_value(mesh)` |
| `gpu_buffer::create`, `image::create`, … | sender → `set_value(...)` |
| `sync_wait_value` / `try_sync_wait_value` | blocking single-value completion |
| `sync_wait` (exceptions ON) | `std::optional<tuple<...>>` — throws on error |
| `sync_wait` / `try_sync_wait` (exceptions OFF) | `sync_wait_outcome<tuple<...>>` |

### Existing SPIR-V (hybrid)

Keep hand-written shaders. vkexec caches the pipeline and records dispatch. Push constants are a host POD (`upload_push_constants` / `compute_pass`):

```cpp
struct ProjectPush { float view[16]; float projection[16]; std::uint64_t gaussian_addr; std::uint32_t splat_count; };

auto pipe = vkexec::sync_wait_value(vkexec::compute_pipeline::create(ctx, glsl, vkexec::layout_desc{
  .bindings = { vkexec::buffer_access::readonly, vkexec::buffer_access::writeonly },
  .push_constant_size = sizeof(ProjectPush),
  .specialization = { splat_count },
  .local_size = { 64, 1, 1 },
}));
// or create(ctx, spirv, layout) when you already have .spv
auto bound = vkexec::sync_wait_value(vkexec::compute_pipeline::bind_storage_sender(pipe, buffers));
if (auto waited = vkexec::sync_wait(ex::schedule(ctx.get_scheduler())
      | vkexec::compute_pass(bound.pipeline, bound.set, push, splat_count));
    !waited.has_value()) { /* stopped */ }
// or, with your own command buffer:
vkexec::upload_push_constants(cmd, pipe, push);
```

### Device requirements and adopt

`vulkan_requirements` is caller-driven: you declare API floors, extensions, and `VkPhysicalDevice*Features` structs; vkexec merges them with a thin library baseline and selects a matching device. Use `context::create` when probing optional capabilities (sender completes with `set_error` / `errc::unsupported` on mismatch).

Embedders that already own a Vulkan device (for example a Filament-like driver) can wrap it without transferring ownership:

```cpp
auto ctx = vkexec::sync_wait_value(vkexec::context::adopt({
  .instance = instance,
  .physical_device = phys,
  .device = device,
  .allocator = vma,              // or null → vkexec creates one
  .compute_queue = compute_q,
  .compute_queue_family = compute_family,
  .graphics_queue = graphics_q,  // optional; defaults to compute
  .present_queue = present_q,    // optional; defaults to graphics
}));
```

**Embedder notes:**

- Destroy the adopted `context` before tearing down the borrowed device/VMA — vkexec owns a command pool and pipeline cache on that device.
- Prefer `context::procs()` over caching `vkGetDeviceProcAddr` results for descriptor-heap / push-data entry points.
- Use `sync_wait_value` / `try_sync_wait_value` for factory senders when you are not composing stdexec graphs.
- With your own command buffers, bindless compute uses `record_heap_pass(ctx, cmd, pipe.bind(), push_bytes, groups)` after `cmd_bind_resource_heap`.
- Keep vk-bootstrap (or your WSI layer) for surface/swapchain when you need app-specific present extensions; use `vkexec_graphics::swapchain` only when a borrowed-surface helper is enough.

When extensions such as `VK_EXT_descriptor_heap` / push-data are enabled, `context::procs()` caches the device PFNs (null when unavailable).

### Bindless (descriptor heap)

Hybrid apps can skip classic descriptor sets. Create a null-layout pipeline with `layout_desc.descriptor_heap = true`, allocate a `gpu_buffer` with `gpu_buffer_memory::descriptor_heap` (optionally `shader_device_address`), write storage descriptors into host-mapped heap memory, then bind + push data on the command buffer:

```cpp
auto layout = vkexec::query_descriptor_heap_layout(ctx);
auto heap = vkexec::sync_wait_value(vkexec::gpu_buffer::create(ctx, {
  .size = vkexec::descriptor_heap_byte_size(layout, slot_count),
  .memory = vkexec::gpu_buffer_memory::descriptor_heap,
  .shader_device_address = true,
}));
vkexec::write_storage_buffer_descriptor(ctx, buffer_addr, buffer_size, heap.mapped().subspan(...));
vkexec::write_storage_image_descriptor(ctx, view_info, VK_IMAGE_LAYOUT_GENERAL, heap.mapped().subspan(...));

auto pipe = vkexec::sync_wait_value(vkexec::compute_pipeline::create(ctx, glsl, vkexec::layout_desc{
  .descriptor_heap = true,
  .push_constant_size = sizeof(Push),
  .local_size = { 64, 1, 1 },
}));

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
./out/build/unixlike-clang-release/src/vkexec_examples/compute
./out/build/unixlike-clang-release/src/vkexec_examples/sort
./out/build/unixlike-clang-release/src/vkexec_examples/passes
./out/build/unixlike-clang-release/src/vkexec_examples/spirv
./out/build/unixlike-clang-release/src/vkexec_examples/triangle
./out/build/unixlike-clang-release/src/vkexec_examples/heap_present
```

`heap_present` is a headless smoke of public Phase 2–4 APIs: optional descriptor-heap compute, dynamic rendering to an offscreen color target, then a few swapchain present frames.

### Triangle window

Requires `vkexec_graphics` (GLFW + swapchain). Shaders are GLSL strings compiled at pipeline creation time. Each frame is a stdexec pipeline:

```cpp
#include <vkexec/sync_wait_outcome.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/triangle_shaders.hpp>
#include <vkexec_graphics/window.hpp>

auto win = vkexec::sync_wait_value(vkexec::window::create({ .width = 800, .height = 600, .title = "triangle" }));

auto pipeline = vkexec::sync_wait_value(vkexec::graphics_pipeline::create(
  win.ctx(), win.render_pass(), vkexec::shaders::k_triangle_vert, vkexec::shaders::k_triangle_frag));

while (!win.should_close()) {
  win.poll_events();
  vkexec::sync_wait(
    ex::schedule(win.ctx().get_scheduler())
    | vkexec::draw(win, pipeline, 3));
}
```


## More Details

 * [Dependency Setup](README_dependencies.md)
 * [Building Details](README_building.md)
 * [Docker](README_docker.md)
