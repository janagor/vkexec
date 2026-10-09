# Compute N-Body

## Origin

Workload inspired by `samples/api/compute_nbody` in
[KhronosGroup/Vulkan-Samples](https://github.com/KhronosGroup/Vulkan-Samples).
The implementation and shaders here are new code; no Khronos assets are copied.

## Workload

Each frame computes pairwise gravitational acceleration for 1,024 particles,
integrates velocity and position, then renders the particles as points:

```text
position -> calculate acceleration -> integrate -> render -> present
```

The two compute passes are in one vkexec pass graph with an explicit
`compute_to_compute` barrier. The graph completes before the graphics sender
starts. Graphics reads the position buffers through storage descriptors.

## Synchronization

The compute graph completes before the graphics submission. The example waits
for the device to become idle after each presented frame so the next frame can
update the same position buffers safely. This serialized path emphasizes the
workload dependency but does not demonstrate multi-queue overlap or frames in
flight.

## Building and running

Configure with `vkexec_BUILD_EXAMPLES=ON`, then build
target `vkexec_vs_compute_nbody`. Run the executable with a Vulkan-capable
display. Close the window to exit.

## Differences from upstream

This is a compact 2D simulation with generated particles and procedural colors.
It does not use upstream textures, particle parameters, or scene data.

## Licensing

See [`../THIRD_PARTY.md`](../THIRD_PARTY.md).
