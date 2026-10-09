# Pipeline Barriers

## Origin

This sample adapts the deferred rendering workload from
[`samples/performance/pipeline_barriers`](https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/performance/pipeline_barriers).
Sponza assets come from Vulkan-Samples-Assets commit
`8db8ce9c528330f0b1261b07531b009732b08731`.

## Workload

The geometry pass draws Sponza with its glTF base-color materials and ASTC KTX
textures into albedo, normal, and depth images. After image barriers, a
full-screen lighting pass samples all three and presents to the swapchain:

```text
Sponza geometry -> albedo, normal, depth -> image barriers -> lighting -> present
```

The example uses vkexec's offscreen target, resource table, graphics pipelines,
image barrier helper, and GLFW presenter. The barrier calls remain visible in
`main.cpp` because the stage choice is the subject of the sample.

## Synchronization

Choose `--barrier=fragment` for color-attachment output to fragment-shader
synchronization, plus depth-test output to fragment-shader synchronization.
Choose `--barrier=conservative` to use `ALL_COMMANDS` for both ends. Both modes
transition the G-buffer images from attachment layouts to
`SHADER_READ_ONLY_OPTIMAL` before lighting. Two offscreen targets alternate with
the presenter's two frames in flight so a target is not rewritten while a prior
lighting pass still samples it.

Use `--view=albedo` to display the RGB albedo G-buffer without lighting. This
helps distinguish missing material color from lighting errors. The supplied
KTX textures use ASTC 8x8 sRGB compression, so the device must support
`textureCompressionASTC_LDR` and sampled ASTC images.

## Building and running

Configure with `vkexec_BUILD_EXAMPLES=ON`, then build
`vkexec_vs_pipeline_barriers`. Run the executable with either barrier mode.
Use WASD to move through Sponza, Q/E to descend/ascend, and right mouse drag to
rotate the view. Close the window to exit; resizing recreates the offscreen targets.

## Differences from upstream

This version uses vkexec's compact glTF loader, material base-color textures,
and starts its free camera at the Main Camera stored in Sponza01.gltf. It uploads the first mip level of
each ASTC KTX texture. It does not reproduce the upstream UI or all of its
material shading controls. The Sponza geometry, G-buffer dependency, and two
barrier modes remain.

## Licensing

The Sponza asset license is in [`assets/sponza/LICENSE.md`](assets/sponza/LICENSE.md).
The example shaders and code are original to vkexec. See
[`../THIRD_PARTY.md`](../THIRD_PARTY.md).
