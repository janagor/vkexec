# Hello Triangle 1.3

## Origin

Workload inspired by `samples/api/hello_triangle_1_3` in
[KhronosGroup/Vulkan-Samples](https://github.com/KhronosGroup/Vulkan-Samples).
This implementation uses vkexec's existing triangle positions and colors, so it
is a workload adaptation rather than a visual reproduction of the upstream sample.

## Workload

One graphics draw of three vertices into the swapchain, followed by presentation.
The application schedules `vkexec::draw` and leaves command submission and
presentation to vkexec.

## Building and running

Configure the project with `vkexec_BUILD_EXAMPLES=ON`,
then build target `vkexec_vs_triangle`. Run the resulting executable with a
Vulkan-capable display. Resize the window or close it to exit.

## Differences from upstream

The shaders use vkexec's existing triangle colors and positions. This sample
uses the repository's GLFW presenter and render-pass path.

## Licensing

The shaders are adaptations of vkexec's existing shaders. See
[`../THIRD_PARTY.md`](../THIRD_PARTY.md).
