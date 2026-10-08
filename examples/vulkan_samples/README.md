# Vulkan Samples workloads in vkexec

This directory is a curated set of vkexec implementations of workloads described by
[KhronosGroup/Vulkan-Samples](https://github.com/KhronosGroup/Vulkan-Samples).
The examples use vkexec to schedule their work and use the existing GLFW presenter
for window and swapchain setup.

| Workload | Status | Main dependency |
| --- | --- | --- |
| Hello Triangle 1.3 | Implemented | graphics to present |
| Compute N-Body | Implemented | compute to compute to graphics |
| Pipeline Barriers | Implemented | G-buffer images to fragment lighting |
| Async Compute | Implemented | graphics and compute queues |
| Timeline Semaphore / Game of Life | Implemented | cross-queue image ping-pong |

The source tree contains GLSL and committed SPIR-V. Normal builds copy the SPIR-V
into the build tree and do not require a shader compiler. Set
`vkexec_BUILD_EXAMPLES=ON` when configuring, then build the
`vkexec_vs_triangle`, `vkexec_vs_compute_nbody`, `vkexec_vs_pipeline_barriers`,
`vkexec_vs_async_compute`, or `vkexec_vs_timeline_semaphore`
target. A Vulkan-capable
display is needed to run them.

When editing shaders, configure with `-Dvkexec_RECOMPILE_SAMPLE_SHADERS=ON` to
compile them into the build tree with `glslc`. To update the committed `.spv`
files, run `bash examples/vulkan_samples/regenerate_shaders.sh` with Shaderc
2026.1. The repository's locked Nix package provides this version:
`nix shell .#sample-shader-compiler --command bash examples/vulkan_samples/regenerate_shaders.sh`.
CI uses the same package to run `validate_shaders.sh` and compare the output
with the committed SPIR-V.

These are workload adaptations, not copies of the Khronos implementation. The
current triangle shaders were adapted from vkexec's existing triangle example.
The N-body shaders and initialization are authored for this suite.
The Pipeline Barriers sample uses Sponza geometry from the Khronos assets
repository; its GLSL is authored here.
The Async Compute sample uses Bonza base-color assets and adapted Khronos GLSL.
The Timeline Semaphore sample uses two runtime images and adapted Khronos GLSL;
it has no external assets.
