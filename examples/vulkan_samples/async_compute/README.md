# Async Compute

This is a vkexec adaptation of the
[Khronos Async Compute workload](https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/performance/async_compute).
It loads Bonza, renders a directional shadow map, renders the scene to an RGBA16F
HDR target, runs threshold and blur passes on a compute queue, and composites HDR
and bloom to the swapchain. The source shaders retain the upstream Apache-2.0
notices; texture and sampler descriptors are separate for vkexec.

## Run

Build target `vkexec_vs_async_compute` with `vkexec_BUILD_EXAMPLES=ON`. Run either:

```sh
vkexec_vs_async_compute --queues=single
vkexec_vs_async_compute --queues=async
```

The default is `--queues=async`. It uses the context's dedicated compute queue
when the graphics and compute families differ and prints the selected families.
On a device without that queue, it falls back to the graphics queue. The `single`
mode always uses the graphics queue as a serialized baseline.

Use WASD and Q/E to move; hold the right mouse button and drag to look around.
The window is resizable. Two per-frame resource sets and binary semaphore pairs
are kept in flight.

## Frame dependencies

```text
graphics: shadow depth -> forward RGBA16F HDR -> signal graphics_done
compute:  wait graphics_done -> threshold -> downsample -> blur up -> signal compute_done
graphics: wait compute_done + image_available -> HDR/bloom composite -> present
```

The shadow image remains on the graphics family. The HDR image transfers to the
compute family after forward rendering and returns to graphics before composition.
The final bloom image transfers to graphics for composition, then returns to
compute when its frame resources are reused. Transfer barriers are emitted only
when queue families differ. Compute passes use sampled input images, storage
output images, and a compute-to-compute image barrier between dependent passes.

The async mode exercises separate queue submission and ownership transfers. It
uses one graphics queue for both early rendering and composition. The upstream
sample can use a second graphics queue so early rendering for the next frame
overlaps the current frame's compute work; that extra queue is outside this
implementation.

The shadow map is 4096² and HDR is twice the swapchain resolution in each axis.
The bloom chain thresholds to half resolution, downsamples to one-sixteenth,
then blurs back up to half resolution. This is deliberately a strong, simple
bloom effect.

See [third-party notices](../THIRD_PARTY.md) for the Bonza scene and shaders.
