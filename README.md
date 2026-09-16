# vkexec

[![ci](https://github.com/janagor/vkexec/actions/workflows/ci.yml/badge.svg)](https://github.com/janagor/vkexec/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/janagor/vkexec/branch/main/graph/badge.svg)](https://codecov.io/gh/janagor/vkexec)
[![CodeQL](https://github.com/janagor/vkexec/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/janagor/vkexec/actions/workflows/codeql-analysis.yml)

## About

`vkexec` is a C++23 **stdexec Vulkan compute backend**: a scheduler, pass/barrier graphs, and borrowable handles — plus optional owning RAII and GLFW helpers for greenfield apps.

**Layer 1 (execution)** — schedule work on a device; record with `compute_bind` / `pipeline_resources`; compose `compute_pass` and barriers. Prefer `#include <vkexec/execution.hpp>`.

**Layer 2 (resources)** — owning `buffer`, `compute_pipeline`, images, samplers. Prefer `#include <vkexec/resources.hpp>` when you want RAII factories. `#include <vkexec/vkexec.hpp>` pulls both.

Same Layer 1 / Layer 2 split applies to optional extensions (descriptor heap, timeline, dynamic rendering).

### Public headers

| Header | Role |
|--------|------|
| `<vkexec/execution.hpp>` | Scheduler, context, pass graphs, `pipeline_resources`, Layer 1 free functions |
| `<vkexec/resources.hpp>` | Owning buffers, images, samplers, `compute_pipeline` |
| `<vkexec/vkexec.hpp>` | Full core umbrella (both layers) |

`context::adopt` borrows instance/device/queues; the `context` still owns a command pool and host/completion agents. Destroy the context (and any vkexec-created resources) before tearing down borrowed Vulkan objects.

### Hero — Layer 1 dispatch

```cpp
#include <vkexec/execution.hpp>
#include <vkexec/resources.hpp>  // optional: typed buffer helpers for the sample

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
    // Layer 2 convenience for host-visible storage (or use your own VkBuffers):
    auto positions = vkexec::detail::take_sync_value(*vkexec::sync_wait(vkexec::buffer<float>::allocate(*ctx, 10000, 0.0f)));
    auto velocities = vkexec::detail::take_sync_value(*vkexec::sync_wait(vkexec::buffer<float>::allocate(*ctx, 10000, 1.5f)));

    using enum vkexec::buffer_access;
    auto resources = vkexec::expected_take(vkexec::create_compute_resources(*ctx, k_sim_glsl,
      vkexec::layout_desc{ .bindings = { readwrite, readwrite }, .push_constant_size = sizeof(sim_params) },
      "sim.comp"));

    std::array const bindings{
      vkexec::storage_binding{ .buffer = positions.vk_buffer(), .byte_size = 10000 * sizeof(float), .binding = 0 },
      vkexec::storage_binding{ .buffer = velocities.vk_buffer(), .byte_size = 10000 * sizeof(float), .binding = 1 },
    };
    auto bound = vkexec::expected_take(vkexec::bind_storage(*ctx, resources, bindings));

    sim_params params{ 0.016f, 0.99f };
    auto graph = ex::schedule(ctx->get_scheduler())
      | vkexec::compute_pass(resources, bound.set, params, 10000)
      | vkexec::barrier::compute_to_compute()
      | vkexec::compute_pass(resources, bound.set, params, 10000);
    if (auto waited = vkexec::sync_wait(std::move(graph)); !waited.has_value()) { return 1; }

    vkexec::free_compute_set(*ctx, resources, bound.set);
    vkexec::destroy_compute_resources(*ctx, resources);
  } catch (vkexec::error const& err) {
    std::cerr << std::format("{}\n", err.message());
    return 1;
  }
}
```

### Greenfield Layer 2 (owning factories)

For apps that want move-only RAII instead of bare `pipeline_resources` (see also [`src/vkexec_examples/compute.cpp`](src/vkexec_examples/compute.cpp)). The Layer 1 path above matches [`src/vkexec_examples/compute_layer1.cpp`](src/vkexec_examples/compute_layer1.cpp).

```cpp
auto pipe = vkexec::detail::take_sync_value(*vkexec::sync_wait(vkexec::compute_pipeline::create(*ctx,
  k_sim_glsl,
  vkexec::layout_desc{ .bindings = { readwrite, readwrite }, .push_constant_size = sizeof(sim_params) },
  "sim.comp")));
auto bound = vkexec::detail::take_sync_value(*vkexec::sync_wait(
  vkexec::bind_storage_sender(pipe, bindings)));
ex::schedule(ctx->get_scheduler()) | vkexec::compute_pass(*bound.pipe, bound.set, params, 10000);
```

### Embedder path — adopt + raw `VkBuffer`s

When you already own the device and buffers, skip Layer 2 factories. Pass borrowed handles into `context::adopt` and `storage_binding`:

```cpp
#include <vkexec/execution.hpp>

auto ctx = vkexec::sync_wait_value(vkexec::context::adopt({
  .instance = instance,
  .physical_device = phys,
  .device = device,
  .allocator = vma,  // or null → vkexec creates one
  .compute_queue = compute_q,
  .compute_queue_family = compute_family,
}));

auto resources = vkexec::expected_take(vkexec::create_compute_resources(*ctx, spirv, layout));
std::array const bindings{
  vkexec::storage_binding{ .buffer = my_positions, .byte_size = bytes, .binding = 0 },
  vkexec::storage_binding{ .buffer = my_velocities, .byte_size = bytes, .binding = 1 },
};
auto bound = vkexec::expected_take(vkexec::bind_storage(*ctx, resources, bindings));

ex::schedule(ctx->get_scheduler())
  | vkexec::compute_pass(resources, bound.set, push, work_count);

// After GPU work finishes:
vkexec::free_compute_set(*ctx, resources, bound.set);
vkexec::destroy_compute_resources(*ctx, resources);
// then destroy ctx before tearing down borrowed device / VMA
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

if (auto buf = vkexec::try_sync_wait_value(vkexec::gpu_buffer::create(*ctx, info)); buf) {
  // use *buf
}
```

`sync_wait_value` throws `vkexec::error` on failure or stop (when exceptions are enabled). `try_sync_wait_value` returns `vkexec::result<T>` instead.

Common entry points:

| API | Returns |
|-----|---------|
| `context::create` / `context::adopt` | sender → `set_value(std::unique_ptr<context>)` |
| `create_compute_resources` / `bind_storage` / `free_compute_set` | Layer 1 classic pipeline + descriptor set loans |
| `buffer<T>::allocate` / `create` | sender → `set_value(buffer<T>)` |
| `compute_pipeline::create` | sender → `set_value(compute_pipeline)` |
| `bind_storage_sender` | sender → `set_value(bound_compute_pipeline)` |
| `window::create` / `window::headless` | sender → `set_value(window)` |
| `graphics_pipeline::create` | sender → `set_value(graphics_pipeline)` |
| `mesh::create` | sender → `set_value(mesh)` |
| `gpu_buffer::create`, `image::create`, … | sender → `set_value(...)` |
| `sync_wait_value` / `try_sync_wait_value` | blocking single-value completion |
| `sync_wait` (exceptions ON) | `std::optional<tuple<...>>` — throws on error |
| `sync_wait` / `try_sync_wait` (exceptions OFF) | `sync_wait_outcome<tuple<...>>` |

### Existing SPIR-V (hybrid)

Keep hand-written shaders. vkexec creates an owning pipeline and records dispatch. Push constants are a host POD (`upload_push_constants` / `compute_pass`):

```cpp
struct ProjectPush { float view[16]; float projection[16]; std::uint64_t gaussian_addr; std::uint32_t splat_count; };

auto pipe = vkexec::sync_wait_value(vkexec::compute_pipeline::create(ctx, glsl, vkexec::layout_desc{
  .bindings = { vkexec::buffer_access::readonly, vkexec::buffer_access::writeonly },
  .push_constant_size = sizeof(ProjectPush),
  .specialization = { splat_count },
  .local_size = { 64, 1, 1 },
}));
// or create(ctx, spirv, layout) when you already have .spv
auto bound = vkexec::sync_wait_value(vkexec::bind_storage_sender(pipe, buffers));
if (auto waited = vkexec::sync_wait(ex::schedule(ctx.get_scheduler())
      | vkexec::compute_pass(*bound.pipe, bound.set, push, splat_count));
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

- Destroy adopted `context` and any owning pipelines before tearing down the borrowed device/VMA — vkexec owns a command pool on that device.
- Use `sync_wait_value` / `try_sync_wait_value` for factory senders when you are not composing stdexec graphs.
- For bindless compute, link `vkexec::ext_descriptor_heap` and use `descriptor_heap_procs_for(ctx)` (or `ext::available<ext::descriptor_heap>(ctx)`) instead of core `context::procs()`.
- With your own command buffers, bindless compute uses `record_heap_pass(ctx, cmd, pipe.bind(), push_bytes, groups)` after `cmd_bind_resource_heap`.
- Keep vk-bootstrap (or your WSI layer) for surface/swapchain when you need app-specific present extensions; use `vkexec_graphics::swapchain` only when a borrowed-surface helper is enough.

Core `context::procs()` exposes only baseline device entry points (e.g. buffer device address). Extension-specific PFNs live in each extension target.

Supporting RAII in core (`<vkexec/resources.hpp>`): `gpu_buffer`, `image` / `image_view` / `sampler`, owning `compute_pipeline`. Optional timeline sync and timeline-based present (`timeline_semaphore`, `frame_ring`, `acquire_present_frame`) live in `vkexec::ext_timeline_semaphore`; see extensions table below. Fence-based present stays in `vkexec_graphics` via `window`.

### Promoted features (`vkexec_features`)

Vulkan capabilities that were promoted from KHR extensions into core versions (or are core-only from some version up) live in [`include/vkexec_features/`](include/vkexec_features/) as **`vkexec::features`**. Use tag types with `feature_traits` via [`feature.hpp`](include/vkexec_features/feature.hpp):

```cpp
#include <vkexec_features/bundles/vulkan_13.hpp>
#include <vkexec_features/feature.hpp>

vkexec::vulkan_requirements req{};
req.api_version_major = 1;
req.api_version_minor = 3;
vkexec::feat::configure_vulkan_13(req);  // timeline + BDA + dynamic rendering
// or: vkexec::feat::configure<vkexec::feat::timeline_semaphore>(req);

auto ctx = vkexec::sync_wait_value(vkexec::context::create({ .requirements = req }));
if (vkexec::feat::available<vkexec::feat::dynamic_rendering>(*ctx)) { /* ... */ }
```

| Feature | Tag | Core version | KHR below core |
|---------|-----|--------------|----------------|
| Timeline semaphore | `feat::timeline_semaphore` | Vulkan 1.2 | `VK_KHR_timeline_semaphore` |
| Buffer device address | `feat::buffer_device_address` | Vulkan 1.2 | `VK_KHR_buffer_device_address` |
| Dynamic rendering | `feat::dynamic_rendering` | Vulkan 1.3 | `VK_KHR_dynamic_rendering` |

`feat::configure` picks the core or KHR path from the requested API version. `feat::available` queries physical-device feature bits on the live context.

### Optional extensions (`vkexec_extensions`)

Core [`vkexec.hpp`](include/vkexec/vkexec.hpp) covers stdexec compute, classic descriptors, buffers, and adopt/create. Optional capability **implementations** (Layer 1 commands + Layer 2 RAII) live under [`include/vkexec_extensions/`](include/vkexec_extensions/) as separate CMake targets — link only what you need. Enable capabilities first with [`vkexec_features`](include/vkexec_features/feature.hpp) (`feat::configure` / `feat::available`).

True Vulkan extensions (`VK_EXT_*`) also expose `ext::configure` / `ext::available` via [`extension.hpp`](include/vkexec_extensions/extension.hpp):

```cpp
#include <vkexec_extensions/descriptor_heap/extension.hpp>
#include <vkexec_extensions/extension.hpp>
#include <vkexec_features/bundles/vulkan_12.hpp>

vkexec::vulkan_requirements req{};
req.api_version_major = 1;
req.api_version_minor = 4;
vkexec::feat::configure_vulkan_12(req);
vkexec::ext::configure<vkexec::ext::descriptor_heap>(req);
auto ctx = vkexec::sync_wait_value(vkexec::context::create({ .requirements = req }));

if (vkexec::ext::available<vkexec::ext::descriptor_heap>(*ctx)) {
  // extension PFNs and helpers are usable
}
```

| Module | CMake target | Include | Gated by |
|--------|--------------|---------|----------|
| Timeline sync | `vkexec::ext_timeline_semaphore` | `<vkexec_extensions/timeline_semaphore.hpp>` | `feat::timeline_semaphore` |
| Descriptor heap | `vkexec::ext_descriptor_heap` | `<vkexec_extensions/descriptor_heap.hpp>` | `ext::descriptor_heap` (+ `feat::buffer_device_address`) |
| Dynamic rendering | `vkexec::ext_dynamic_rendering` | `<vkexec_extensions/dynamic_rendering.hpp>` | `feat::dynamic_rendering` |

**Layer 1 — free functions** (adopt-everything embedders; same idea as core `<vkexec/execution.hpp>`): proc lookup (`descriptor_heap_procs_for`), descriptor writes (storage buffer/image, sampled image, sampler), `cmd_bind_resource_heap`, `cmd_bind_sampler_heap`, `cmd_push_data`, `record_heap_pass`, `cmd_begin_rendering`, etc.

**Layer 2 — RAII types** (vkexec-native apps; same idea as core `<vkexec/resources.hpp>`): `timeline_semaphore`, `frame_ring`, `acquire_present_frame` / `submit_and_present`, `descriptor_heap_buffer`, `heap_compute_pipeline`, `compute_heap_pass`, rendering helpers built on top of Layer 1.

**Timeline sync:** enable with `feat::configure<feat::timeline_semaphore>` (or `configure_vulkan_12`), link `vkexec::ext_timeline_semaphore`, then create semaphores, a present frame ring, or timeline-based acquire/present:

```cpp
#include <vkexec_features/timeline_semaphore.hpp>
#include <vkexec_extensions/timeline_semaphore.hpp>

vkexec::feat::configure<vkexec::feat::timeline_semaphore>(req);
auto sem = vkexec::sync_wait_value(vkexec::timeline_semaphore::create(*ctx, 0));
// optional: acquire_present_frame(ring, chain, slot) / submit_and_present(...)
```

**Descriptor heap (bindless):** link `vkexec::ext_descriptor_heap`, create a null-layout pipeline with `heap_compute_pipeline`, allocate a `descriptor_heap_buffer`, write descriptors into host-mapped heap memory, then bind + push data:

```cpp
#include <vkexec_extensions/descriptor_heap.hpp>

auto layout = vkexec::query_descriptor_heap_layout(ctx);
auto heap = vkexec::sync_wait_value(
  vkexec::descriptor_heap_buffer::create(ctx, vkexec::descriptor_heap_byte_size(layout, slot_count)));
vkexec::write_storage_buffer_descriptor(ctx, buffer_addr, buffer_size, heap.mapped().subspan(...));
// also: write_sampled_image_descriptor / write_sampler_descriptor into resource / sampler heaps

auto pipe = vkexec::sync_wait_value(vkexec::heap_compute_pipeline::create(ctx, glsl,
  vkexec::heap_layout_desc{ .specialization = {}, .local_size = vkexec::k_default_local_size }));

// stdexec path:
ex::schedule(ctx.get_scheduler()) | vkexec::compute_heap_pass(pipe, push, work_count);

// manual command buffer:
vkexec::cmd_bind_resource_heap(ctx, cmd, heap.device_address(), heap.size(),
  reserved_offset, layout.min_resource_heap_reserved_range);
// also: cmd_bind_sampler_heap(...) for a sampler heap sized with sampler_heap_byte_size
vkexec::cmd_push_data(ctx, cmd, push);
```

Example: [`src/vkexec_examples/extensions/descriptor_heap/`](src/vkexec_examples/extensions/descriptor_heap/) runs a bindless compute dispatch when the extension is available.

**Dynamic rendering:** enable with `feat::configure<feat::dynamic_rendering>` (or `configure_vulkan_13`), then use Layer 1 helpers from the extensions target:

```cpp
#include <vkexec_features/dynamic_rendering.hpp>
#include <vkexec_extensions/dynamic_rendering/rendering.hpp>

vkexec::feat::configure<vkexec::feat::dynamic_rendering>(req);
vkexec::cmd_begin_rendering(cmd, vkexec::rendering_info{ .extent = { w, h }, .color = color_attachments });
vkexec::cmd_end_rendering(cmd);
```

Example: [`src/vkexec_examples/extensions/dynamic_rendering/`](src/vkexec_examples/extensions/dynamic_rendering/) clears the swapchain each frame with dynamic rendering (`vkexec::features` + `vkexec::ext_dynamic_rendering` + `vkexec_graphics` for the window).

Build and run the sample:

```bash
nix develop
cmake --preset unixlike-clang-release
cmake --build out/build/unixlike-clang-release -j12
./out/build/unixlike-clang-release/src/vkexec_examples/compute
./out/build/unixlike-clang-release/src/vkexec_examples/compute_layer1
./out/build/unixlike-clang-release/src/vkexec_examples/sort
./out/build/unixlike-clang-release/src/vkexec_examples/passes
./out/build/unixlike-clang-release/src/vkexec_examples/spirv
./out/build/unixlike-clang-release/src/vkexec_examples/triangle
./out/build/unixlike-clang-release/src/vkexec_examples/heap_present
./out/build/unixlike-clang-release/src/vkexec_examples/extensions/descriptor_heap/descriptor_heap
./out/build/unixlike-clang-release/src/vkexec_examples/extensions/dynamic_rendering/dynamic_rendering
```

`extensions/descriptor_heap` runs a bindless compute smoke test when `VK_EXT_descriptor_heap` is available.

`extensions/dynamic_rendering` opens a window and presents an animated color clear each frame via dynamic rendering.

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
