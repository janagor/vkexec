# Third-party material

The workload names and progression refer to
[KhronosGroup/Vulkan-Samples](https://github.com/KhronosGroup/Vulkan-Samples).
No Khronos source, shaders, or assets are included in the current triangle or
N-body samples. The triangle shaders were adapted from vkexec's existing
triangle shaders; the N-body shaders are new code in this repository.

For any additional upstream shaders or assets, record the exact upstream commit
and retain the applicable copyright notices and asset license files here.

The Pipeline Barriers sample includes `assets/sponza` from
[KhronosGroup/Vulkan-Samples-Assets](https://github.com/KhronosGroup/Vulkan-Samples-Assets)
at commit `8db8ce9c528330f0b1261b07531b009732b08731`. Sponza is licensed
under CC BY 3.0; its license and attribution are preserved in
[`pipeline_barriers/assets/sponza/LICENSE.md`](pipeline_barriers/assets/sponza/LICENSE.md).

The Async Compute sample includes Bonza's glTF, geometry buffer, and base-color
textures from [KhronosGroup/Vulkan-Samples-Assets](https://github.com/KhronosGroup/Vulkan-Samples-Assets)
at commit `8db8ce9c528330f0b1261b07531b009732b08731`. Bonza is Apache-2.0;
its license is retained in
[`async_compute/assets/bonza/LICENSE.md`](async_compute/assets/bonza/LICENSE.md).
The sample shaders are adapted from
[KhronosGroup/Vulkan-Samples](https://github.com/KhronosGroup/Vulkan-Samples)
at commit `177edebf0cd7d4f669667e49f052cfb56b17e004` and retain their
Apache-2.0 headers. Their descriptors and geometry inputs were changed for vkexec.
