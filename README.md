# vkexec

[![ci](https://github.com/janagor/vkexec/actions/workflows/ci.yml/badge.svg)](https://github.com/janagor/vkexec/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/janagor/vkexec/branch/main/graph/badge.svg)](https://codecov.io/gh/janagor/vkexec)
[![CodeQL](https://github.com/janagor/vkexec/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/janagor/vkexec/actions/workflows/codeql-analysis.yml)

## About

`vkexec` is a C++23 **stdexec Vulkan compute backend**: a scheduler, pass/barrier graphs, borrowable handles, and backend-neutral Vulkan presentation helpers.

**Execution** — schedule work on a device; record with `compute_bind` / `handles::compute_pipeline`; compose `compute_pass` and barriers. Prefer `#include <vkexec/execution.hpp>`.

**Resources** — owning `owned::buffer`, typed `owned::tensor<T>`, `owned::compute_pipeline`, images, and samplers. Prefer `#include <vkexec/resources.hpp>` when you want RAII factories. `#include <vkexec/vkexec.hpp>` pulls both.

The same execution / resources split applies to optional extensions (descriptor heap, timeline, dynamic rendering).

### Ownership

Every public type that carries Vulkan or VMA handles is one of two typed lanes:

- `vkexec::handles::` is a non-owning bag of raw handles. The caller must finish GPU work and call `vkexec::destroy(ctx, handles)` (or the equivalent destroy overload); individual fields must not be destroyed while the bag is live.
- `vkexec::owned::` is move-only RAII. Its destructor destroys the handles it owns.

Pass-graph and binding types such as `compute_bind`, `storage_binding`, `resource_ref`, and schedulers are ephemeral borrows. They never destroy what they name and must not outlive those objects. `context` always owns its command pool and host/completion agents; `owns_instance()`, `owns_device()`, and `owns_allocator()` report whether it also owns the corresponding Vulkan/VMA objects.

The context outlives every `owned::` or `handles::` object created against it. Tear down in this order: finish GPU work, destroy owned objects or handle bags, destroy the context, then let the embedder destroy adopted instance/device/VMA objects.

### Public headers

| Header | Role |
|--------|------|
| `<vkexec/execution.hpp>` | Scheduler, context, pass graphs, `handles::compute_pipeline`, free functions |
| `<vkexec/resources.hpp>` | `owned::` buffers, tensors, images, samplers, and pipelines |
| `<vkexec/vkexec.hpp>` | Full core umbrella (execution + resources) |

`factory::adopt_context` borrows instance/device/queues; the `context` still owns a command pool and host/completion agents. Destroy the context (and any vkexec-created resources) before tearing down borrowed Vulkan objects.

### Hero — borrowable dispatch

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
    auto ctx = vkexec::sync_wait_value(vkexec::factory::make_context());
    // Owning buffer helpers for host-visible storage (or use your own VkBuffers):
    auto positions = vkexec::sync_wait_value(vkexec::factory::make_buffer(*ctx, 10000, 0.0f));
    auto velocities = vkexec::sync_wait_value(vkexec::factory::make_buffer(*ctx, 10000, 1.5f));

    using enum vkexec::buffer_access;
    auto resources = vkexec::expected_take(vkexec::create(*ctx, k_sim_glsl,
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
    vkexec::destroy(*ctx, resources);
  } catch (vkexec::error const& err) {
    std::cerr << std::format("{}\n", err.message());
    return 1;
  }
}
```

### Greenfield owning factories

For apps that want move-only RAII instead of bare `handles::compute_pipeline`, use `owned::compute_pipeline` (see also [`examples/compute.cpp`](examples/compute.cpp)). The borrowable path above matches [`examples/compute_execution.cpp`](examples/compute_execution.cpp).

```cpp
auto pipe = vkexec::sync_wait_value(vkexec::factory::make_compute_pipeline(*ctx,
  k_sim_glsl,
  vkexec::layout_desc{ .bindings = { readwrite, readwrite }, .push_constant_size = sizeof(sim_params) },
  "sim.comp"));
auto bound = vkexec::sync_wait_value(vkexec::bind_storage_sender(pipe, bindings));
ex::schedule(ctx->get_scheduler()) | vkexec::compute_pass(*bound.pipe, bound.set, params, 10000);
```

### Staging tensors

Staging-backed `tensor<T>` plus `sync_to_device` / `sync_to_host` pipeables give an upload -> dispatch -> download shape on the same pass graph. Device storage is created with `shader_device_address`, so the context needs `bufferDeviceAddress` (e.g. `feat::configure<feat::buffer_device_address>`). Runnable sample: [`examples/tensor_sim.cpp`](examples/tensor_sim.cpp).

```cpp
#include <vkexec/execution.hpp>
#include <vkexec/resources.hpp>
#include <vkexec_features/buffer_device_address.hpp>

vkexec::feat::configure<vkexec::feat::buffer_device_address>(req);
// ... create context with req ...

auto positions = vkexec::sync_wait_value(vkexec::factory::make_tensor(*ctx, 10000, 0.0f));
auto velocities = vkexec::sync_wait_value(vkexec::factory::make_tensor(*ctx, 10000, 1.5f));
// ... create + bind_storage using positions.storage_binding(0), ...

vkexec::sync_wait(
  ex::schedule(ctx->get_scheduler())
  | vkexec::sync_to_device(positions)
  | vkexec::sync_to_device(velocities)
  | vkexec::compute_pass(resources, bound.set, params, 10000)
  | vkexec::sync_to_host(positions)
  | vkexec::sync_to_host(velocities));
```

For descriptor-heap algorithms, select the strategy once with `descriptor_heap` and use `factory::make_compute_pipeline`, `compute_pass`, or `dispatch_compute`. The longer `dispatch_compute` name avoids colliding with `struct dispatch`.

### Embedder path — adopt + raw `VkBuffer`s

When you already own the device and buffers, skip owning factories. Pass borrowed handles into `factory::adopt_context` and `storage_binding`:

```cpp
#include <vkexec/execution.hpp>

auto ctx = vkexec::sync_wait_value(vkexec::factory::adopt_context({
  .instance = instance,
  .physical_device = phys,
  .device = device,
  .allocator = vma,  // or null -> vkexec creates one
  .compute_queue = compute_q,
  .compute_queue_family = compute_family,
}));

auto resources = vkexec::expected_take(vkexec::create(*ctx, spirv, layout));
std::array const bindings{
  vkexec::storage_binding{ .buffer = my_positions, .byte_size = bytes, .binding = 0 },
  vkexec::storage_binding{ .buffer = my_velocities, .byte_size = bytes, .binding = 1 },
};
auto bound = vkexec::expected_take(vkexec::bind_storage(*ctx, resources, bindings));

ex::schedule(ctx->get_scheduler())
  | vkexec::compute_pass(resources, bound.set, push, work_count);

// After GPU work finishes:
vkexec::free_compute_set(*ctx, resources, bound.set);
vkexec::destroy(*ctx, resources);
// then destroy ctx before tearing down borrowed device / VMA
```

### Error model

Public APIs are **senders** (stdexec). Completions follow stdexec semantics:

- **`set_value(...)`** — success
- **`set_error(vkexec::error)`** — failure (same error type everywhere async)
- **`set_stopped()`** — cancellation (not an error)

Factory functions such as `factory::make_context`, `factory::make_buffer`, and `factory::make_compute_pipeline` return `vkexec::sender<T>`; compose them with `stdexec::let_value` or block at the sync boundary with `sync_wait_value`.

- **`vkexec::error`** carries a `boost::system::error_code` plus optional detail text. Use `error.message()` for a human-readable string.
- **`vkexec::errc`** covers library-level failures (`invalid_argument`, `unsupported`, `cancelled`, …).
- **Vulkan failures** use `vkexec::make_vk_error_code(VkResult)` / `vkexec::make_vk_error(...)`.

**Sync boundary (`VKEXEC_ENABLE_EXCEPTIONS`, default ON):** `vkexec::sync_wait(sender)` delegates to `stdexec::sync_wait` and **throws `vkexec::error`** on `set_error`. A disengaged `std::optional` means `set_stopped()` (not an error).

```cpp
try {
  auto ctx = vkexec::sync_wait_value(vkexec::factory::make_context({ .requirements = reqs }));
  auto buf = vkexec::sync_wait_value(vkexec::factory::make_buffer(*ctx, n, fill));
} catch (vkexec::error const &err) {
  std::cout << std::format("{}\n", err.message());
}
```

**`-fno-exceptions` builds (`-DVKEXEC_ENABLE_EXCEPTIONS=OFF`):** `sync_wait` returns `vkexec::sync_wait_outcome<...>` with `values`, `error`, and `stopped` fields — no throwing, no `std::expected`.

```cpp
auto outcome = vkexec::try_sync_wait_value(vkexec::factory::make_context({ .requirements = reqs }));
if (!outcome) {
  std::cout << std::format("{}\n", outcome.error().message());
  return;
}
auto ctx = vkexec::expected_take(outcome);
```

For tests without exceptions, use `vkexec::try_sync_wait` (same `sync_wait_outcome` shape).

Embedders that do not use stdexec pipes can block on factory senders directly:

```cpp
auto ctx = vkexec::sync_wait_value(vkexec::factory::adopt_context({ /* borrowed handles */ }));
auto pipe = vkexec::sync_wait_value(vkexec::factory::make_compute_pipeline(*ctx, spirv, layout));

if (auto buf = vkexec::try_sync_wait_value(vkexec::factory::make_gpu_buffer(*ctx, info)); buf) {
  // use *buf
}
```

`sync_wait_value` throws `vkexec::error` on failure or stop (when exceptions are enabled). `try_sync_wait_value` returns `vkexec::result<T>` instead.

Common entry points:

| API | Returns |
|-----|---------|
| `factory::make_context` / `factory::adopt_context` | sender -> `set_value(std::unique_ptr<context>)` |
| `create` / `bind_storage` / `free_compute_set` | Borrowable classic pipeline + descriptor set loans |
| `create(descriptor_heap, …)` / `bind_compute` / `destroy` | Borrowable descriptor-heap compute pipeline bags (`vkexec::ext_descriptor_heap`) |
| `create(descriptor_heap, …)` / `bind_compute` / `destroy(descriptor_heap, …)` | Borrowable descriptor-heap graphics (DR formats, null layout; compose with `cmd_begin_rendering`) |
| `factory::make_buffer` | sender -> `set_value(owned::buffer<T>)` |
| `factory::make_compute_pipeline` | sender -> `set_value(owned::compute_pipeline)` |
| `bind_storage_sender` | sender -> `set_value(bound_compute_pipeline)` |
| `factory::make_presenter` / `factory::make_headless_presenter` | sender -> `set_value(owned::presenter)` (owning Vulkan present helper) |
| `create` / `bind_graphics_storage` / `free_graphics_set` | Borrowable classic graphics pipeline + descriptor set loans |
| `create` / `destroy` | Borrowable vertex/index handle bag |
| `factory::make_graphics_pipeline` | sender -> `set_value(owned::graphics_pipeline)` (thin owning wrapper) |
| `factory::make_mesh` | sender -> `set_value(owned::mesh)` (thin owning wrapper) |
| `factory::make_gpu_buffer`, `factory::make_image`, … | sender -> `set_value(...)` |
| `sync_wait_value` / `try_sync_wait_value` | blocking single-value completion |
| `sync_wait` (exceptions ON) | `std::optional<tuple<...>>` — throws on error |
| `sync_wait` / `try_sync_wait` (exceptions OFF) | `sync_wait_outcome<tuple<...>>` |

### Existing SPIR-V (hybrid)

Keep hand-written shaders. vkexec creates an owning pipeline and records dispatch. Push constants are a host POD (`upload_push_constants` / `compute_pass`):

```cpp
struct ProjectPush { float view[16]; float projection[16]; std::uint64_t gaussian_addr; std::uint32_t splat_count; };

auto pipe = vkexec::sync_wait_value(vkexec::factory::make_compute_pipeline(ctx, glsl, vkexec::layout_desc{
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

`vulkan_requirements` is caller-driven: you declare API floors, extensions, and `VkPhysicalDevice*Features` structs; vkexec merges them with a thin library baseline and selects a matching device. Use `factory::make_context` when probing optional capabilities (sender completes with `set_error` / `errc::unsupported` on mismatch).

Embedders that already own a Vulkan device (for example a Filament-like driver) can wrap it without transferring ownership:

```cpp
auto ctx = vkexec::sync_wait_value(vkexec::factory::adopt_context({
  .instance = instance,
  .physical_device = phys,
  .device = device,
  .allocator = vma,              // or null -> vkexec creates one
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
- With your own command buffers, descriptor-heap compute uses `record_pass(ctx, cmd, pipe.bind(), push_bytes, groups)` after `cmd_bind_resource_heap`.
- Use `presenter` when vkexec should own the presentation context and surface; use `swapchain` when an embedder already owns its Vulkan context and surface.

Core `context::procs()` exposes only baseline device entry points (e.g. buffer device address). Extension-specific PFNs live in each extension target.

Supporting RAII in core (`<vkexec/resources.hpp>`): `gpu_buffer`, staging-backed `tensor<T>`, `image` / `image_view` / `sampler`, owning `compute_pipeline`. Optional timeline sync and timeline-based present (`timeline_semaphore`, `frame_ring`, `acquire_present_frame`) live in `vkexec::ext_timeline_semaphore`; see extensions table below. Fence-based present stays in `vkexec_graphics` via `presenter`.

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

auto ctx = vkexec::sync_wait_value(vkexec::factory::make_context({ .requirements = req }));
if (vkexec::feat::available<vkexec::feat::dynamic_rendering>(*ctx)) { /* ... */ }
```

| Feature | Tag | Core version | KHR below core |
|---------|-----|--------------|----------------|
| Timeline semaphore | `feat::timeline_semaphore` | Vulkan 1.2 | `VK_KHR_timeline_semaphore` |
| Buffer device address | `feat::buffer_device_address` | Vulkan 1.2 | `VK_KHR_buffer_device_address` |
| Dynamic rendering | `feat::dynamic_rendering` | Vulkan 1.3 | `VK_KHR_dynamic_rendering` |

`feat::configure` picks the core or KHR path from the requested API version. `feat::available` queries physical-device feature bits on the live context.

### Optional extensions (`vkexec_extensions`)

Core [`vkexec.hpp`](include/vkexec/vkexec.hpp) covers stdexec compute, classic descriptors, buffers, and adopt/create. Optional capability **implementations** (command free functions + owning RAII) live under [`include/vkexec_extensions/`](include/vkexec_extensions/) as separate CMake targets — link only what you need. Enable capabilities first with [`vkexec_features`](include/vkexec_features/feature.hpp) (`feat::configure` / `feat::available`).

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
auto ctx = vkexec::sync_wait_value(vkexec::factory::make_context({ .requirements = req }));

if (vkexec::ext::available<vkexec::ext::descriptor_heap>(*ctx)) {
  // extension PFNs and helpers are usable
}
```

| Module | CMake target | Include | Gated by |
|--------|--------------|---------|----------|
| Timeline sync | `vkexec::ext_timeline_semaphore` | `<vkexec_extensions/timeline_semaphore.hpp>` | `feat::timeline_semaphore` |
| Descriptor heap | `vkexec::ext_descriptor_heap` | `<vkexec_extensions/descriptor_heap.hpp>` | `ext::descriptor_heap` (+ `feat::buffer_device_address`) |
| Dynamic rendering | `vkexec::ext_dynamic_rendering` | `<vkexec_extensions/dynamic_rendering.hpp>` | `feat::dynamic_rendering` |

**Free functions** (adopt-everything embedders; same idea as core `<vkexec/execution.hpp>`): proc lookup (`descriptor_heap_procs_for`), descriptor writes (storage buffer/image, sampled image, sampler), tagged `create` / `create`, `bind_compute`, tagged graphics destroy, `cmd_bind_resource_heap`, `cmd_bind_sampler_heap`, `cmd_push_data`, `record_pass`, `record_draw` / `record_draw_indirect`, `cmd_begin_rendering`, etc.

**Owning RAII types** (vkexec-native apps; same idea as core `<vkexec/resources.hpp>`): `owned::timeline_semaphore`, `owned::frame_ring`, `acquire_present_frame` / `submit_and_present`, `descriptor_heap_buffer`, tagged `factory::make_compute_pipeline` / `factory::make_graphics_pipeline`, `compute_pass`, and rendering helpers built on top of the free functions.

**Timeline sync:** enable with `feat::configure<feat::timeline_semaphore>` (or `configure_vulkan_12`), link `vkexec::ext_timeline_semaphore`, then create semaphores, a present frame ring, or timeline-based acquire/present:

```cpp
#include <vkexec_features/timeline_semaphore.hpp>
#include <vkexec_extensions/timeline_semaphore.hpp>

vkexec::feat::configure<vkexec::feat::timeline_semaphore>(req);
auto sem = vkexec::sync_wait_value(vkexec::factory::make_timeline_semaphore(*ctx, 0));
// optional: acquire_present_frame(ring, chain, slot) / submit_and_present(...)
```

**Descriptor heap (bindless):** link `vkexec::ext_descriptor_heap`, create a null-layout pipeline by passing `descriptor_heap` to the regular factory verb, allocate a `descriptor_heap_buffer`, write descriptors into host-mapped heap memory, then bind + push data. Heap **graphics** pipelines use dynamic-rendering formats (`VkPipelineRenderingCreateInfo`) with no `VkRenderPass`; compose with `cmd_begin_rendering` + `record_draw(ctx, …)` (classic `vkexec_graphics` stays render-pass based).

```cpp
#include <vkexec_extensions/descriptor_heap.hpp>

auto layout = vkexec::query_descriptor_heap_layout(ctx);
auto heap = vkexec::sync_wait_value(
  vkexec::factory::make_descriptor_heap_buffer(ctx, vkexec::descriptor_heap_byte_size(layout, slot_count)));
vkexec::write_storage_buffer_descriptor(ctx, buffer_addr, buffer_size, heap.mapped().subspan(...));
// also: write_sampled_image_descriptor / write_sampler_descriptor into resource / sampler heaps

auto resources = vkexec::expected_take(vkexec::create(vkexec::descriptor_heap, ctx, glsl,
  vkexec::heap_layout_desc{ .specialization = {}, .local_size = vkexec::k_default_local_size }));
// or owning: factory::make_compute_pipeline(descriptor_heap, ...)

using schema = vkexec::descriptor_schema<vkexec::storage_buffer<0>>;
auto table = vkexec::make_resource_table(schema{}, vkexec::buffer_resource(buffer, buffer_size));
vkexec::heap_table_lower_env heap_env{ /* mapped bytes, descriptor sizes, indices */ };

// stdexec path: lower + bind + push, then dispatch.
ex::schedule(ctx.get_scheduler())
  | vkexec::bind_resources(vkexec::descriptor_heap, resources, table, heap_env, push)
  | vkexec::compute_pass(vkexec::bind_compute(resources), groups);

// manual command buffer:
vkexec::cmd_bind_resource_heap(ctx, cmd, heap.device_address(), heap.size(),
  reserved_offset, layout.min_resource_heap_reserved_range);
// also: cmd_bind_sampler_heap(...) for a sampler heap sized with sampler_heap_byte_size
vkexec::cmd_push_data(ctx, cmd, push);
// peel VkPipeline with resources.pipeline when recording yourself

vkexec::destroy(ctx, resources);
```

`resource_table` and `descriptor_schema` also support storage images, sampled images, and samplers (`storage_image`, `sampled_image`, and `sampler_binding`). Descriptor-set lowering consumes Vulkan handles directly. Descriptor-heap lowering keeps physical resource/sampler indices, mapped heap spans, strides, and required image/sampler create infos in the extension-only `heap_table_lower_env`; vkexec does not allocate heap slots or emulate descriptor sets. Literal heap object APIs (`descriptor_heap_buffer`, `cmd_bind_*_heap`, `write_*_descriptor`) remain unchanged.

For classic descriptors, `schema_pass` collapses table construction, descriptor binding, push data, and dispatch into one typed pipeable while keeping the individual steps available:

```cpp
using schema = vkexec::descriptor_schema<vkexec::storage_buffer<0>, vkexec::storage_buffer<1>>;

ex::schedule(ctx->get_scheduler())
  | vkexec::schema_pass(schema{}, resources, params, work_count, positions, velocities);

// Borrowable lower-level path:
auto table = vkexec::make_resource_table(schema{}, positions_ref, velocities_ref);
ex::schedule(ctx->get_scheduler())
  | vkexec::bind_resources(resources, table, params)
  | vkexec::compute_pass(vkexec::bind_compute(resources), groups);
```

Example: [`examples/extensions/descriptor_heap/`](examples/extensions/descriptor_heap/) runs a bindless compute dispatch when the extension is available.

**Dynamic rendering:** enable with `feat::configure<feat::dynamic_rendering>` (or `configure_vulkan_13`), then use free-function helpers from the extensions target:

```cpp
#include <vkexec_features/dynamic_rendering.hpp>
#include <vkexec_extensions/dynamic_rendering/rendering.hpp>

vkexec::feat::configure<vkexec::feat::dynamic_rendering>(req);
vkexec::cmd_begin_rendering(cmd, vkexec::rendering_info{ .extent = { w, h }, .color = color_attachments });
vkexec::cmd_end_rendering(cmd);
```

Example: [`examples/extensions/dynamic_rendering/`](examples/extensions/dynamic_rendering/) clears the swapchain each frame with dynamic rendering (`vkexec::features` + `vkexec::ext_dynamic_rendering` + `vkexec_graphics` for the window).

Build and run the sample:

```bash
nix develop
cmake --preset unixlike-clang-release
cmake --build out/build/unixlike-clang-release -j12
./out/build/unixlike-clang-release/examples/compute
./out/build/unixlike-clang-release/examples/compute_execution
./out/build/unixlike-clang-release/examples/tensor_sim
./out/build/unixlike-clang-release/examples/sort
./out/build/unixlike-clang-release/examples/passes
./out/build/unixlike-clang-release/examples/spirv
./out/build/unixlike-clang-release/examples/triangle
./out/build/unixlike-clang-release/examples/graphics_execution
./out/build/unixlike-clang-release/examples/heap_present
./out/build/unixlike-clang-release/examples/extensions/descriptor_heap/descriptor_heap
./out/build/unixlike-clang-release/examples/extensions/dynamic_rendering/dynamic_rendering
```

`extensions/descriptor_heap` runs a bindless compute smoke test when `VK_EXT_descriptor_heap` is available.

`extensions/dynamic_rendering` opens a window and presents an animated color clear each frame via dynamic rendering.

`heap_present` is a headless smoke of public Phase 2–4 APIs: optional descriptor-heap compute and heap graphics (DR + `record_draw`), dynamic rendering clear to an offscreen color target, then a few swapchain present frames.

### Graphics execution / resources

`vkexec_graphics` mirrors the core split:

| Header | Role |
|--------|------|
| `<vkexec_graphics/execution.hpp>` | `handles::graphics_pipeline`, `handles::mesh`, bind/record/draw free functions, swapchain (borrow surface) |
| `<vkexec_graphics/resources.hpp>` | Owning `owned::presenter`, `owned::graphics_pipeline`, `owned::mesh` |
| `<vkexec_graphics/vkexec_graphics.hpp>` | Full graphics umbrella (execution + resources) |

`owned::presenter` owns its Vulkan context, surface, swapchain, and frame resources without owning a native window or event loop. `owned::swapchain` borrows an embedder-owned context and surface. Pipeline create/bind/record is borrowable; `owned::graphics_pipeline` / `owned::mesh` are thin RAII wrappers.

### User-owned window and surface factory

The application supplies its window-system extensions and a callback that creates `VkSurfaceKHR` after vkexec creates the Vulkan instance. The presenter owns the returned surface; the native window must outlive the presenter. GLFW is used only by the examples:

```cpp
std::uint32_t extension_count = 0;
auto extensions = glfwGetRequiredInstanceExtensions(&extension_count);
std::vector<char const*> surface_extensions(extensions, extensions + extension_count);

auto present = vkexec::sync_wait_value(vkexec::factory::make_presenter({
  .width = 800,
  .height = 600,
  .surface_instance_extensions = std::move(surface_extensions),
  .create_surface = [native_window](VkInstance instance) -> vkexec::result<VkSurfaceKHR> {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkResult result = glfwCreateWindowSurface(instance, native_window, nullptr, &surface);
    if (result != VK_SUCCESS) { return vkexec::fail(result, "glfwCreateWindowSurface failed"); }
    return surface;
  },
}));

auto pipeline = vkexec::sync_wait_value(vkexec::factory::make_graphics_pipeline(
  present.ctx(), present.render_pass(), vertex_shader, fragment_shader));

while (!glfwWindowShouldClose(native_window)) {
  glfwPollEvents();
  if (framebuffer_resized || present.needs_resize()) {
    glfwGetFramebufferSize(native_window, &width, &height);
    present.resize(width, height); // zero extent suspends presentation while minimized
  }
  vkexec::sync_wait(
    ex::schedule(present.ctx().get_scheduler())
    | vkexec::draw(present, pipeline, 3));
}
```

See [`examples/triangle.cpp`](examples/triangle.cpp).

### Graphics execution — borrowed pipeline handles

Same triangle present path without an owning `graphics_pipeline`. Prefer `#include <vkexec_graphics/execution.hpp>`:

```cpp
#include <vkexec_graphics/execution.hpp>
#include <vkexec_graphics/triangle_shaders.hpp>
// `present` is an owned::presenter created from the application's WSI callback.

auto resources = vkexec::expected_take(vkexec::create(
  present.ctx(), present.render_pass(), {}, vkexec::shaders::k_triangle_vert, vkexec::shaders::k_triangle_frag));

while (application_is_running) {
  vkexec::sync_wait(
    ex::schedule(present.ctx().get_scheduler())
    | vkexec::draw(present, resources, VK_NULL_HANDLE, 3));
}

present.wait_idle();
vkexec::destroy(present.ctx(), resources);
```

Runnable sample: [`examples/graphics_execution.cpp`](examples/graphics_execution.cpp).

### Embedder path — raw `VkBuffer` storage bindings

When shaders declare storage buffers, create resources with a binding count, then loan a set from raw handles (same contract as compute `bind_storage`):

```cpp
auto resources = vkexec::expected_take(vkexec::create(
  ctx, render_pass, cfg, vert_spirv, frag_spirv, /*storage_binding_count=*/1));

std::array const bindings{
  vkexec::storage_binding{ .buffer = my_ssbo, .byte_size = bytes, .binding = 0 },
};
auto bound = vkexec::expected_take(vkexec::bind_graphics_storage(ctx, resources, bindings));

ex::schedule(ctx.get_scheduler())
  | vkexec::draw(win, resources, bound.set, vertex_count);

// After GPU work finishes:
vkexec::free_graphics_set(ctx, resources, bound.set);
vkexec::destroy(ctx, resources);
```

Mesh draws take a `mesh_draw` / `handles::mesh` handle bag the same way — owning `owned::mesh` is optional.


## More Details

 * [Dependency Setup](README_dependencies.md)
 * [Building Details](README_building.md)
 * [Docker](README_docker.md)
