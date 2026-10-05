# Timeline Semaphore / Game of Life

This sample runs the [Khronos Timeline Semaphore workload](https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/extensions/timeline_semaphore)
with vkexec's queue submission, image allocation, presenter, and timeline semaphore
APIs. It uses the five upstream shaders from commit
`177edebf0cd7d4f669667e49f052cfb56b17e004`, with only texture and sampler
descriptor declarations adapted for vkexec. There are no external assets.

Build `vkexec_vs_timeline_semaphore` with `vkexec_BUILD_EXAMPLES=ON` and `glslc`
available. The sample requires Vulkan 1.3 and the `timelineSemaphore` feature.
Run it on a Vulkan-capable display; close the window to exit.

## Images and queues

Two 64×64 `R8G8B8A8_UNORM` images alternate roles. The original Khronos init
shader fills both images at startup. Each frame, compute samples the previous
image and writes the other one; graphics samples the new image and draws a
fullscreen triangle. Nearest filtering keeps the cells visibly pixelated.

```text
Image A ──sample──> compute queue ──write──> Image B
                                             │
                                      timeline draw(N)
                                             │
                                             ▼
                                        graphics queue
                                             │
                                             ▼
                                          swapchain
```

The images use concurrent sharing when graphics and compute use different queue
families. They use exclusive sharing when both queues have the same family.
The compute worker uses the original update shader after a one-second interval;
between updates it uses the original mutate shader and its float push constant
to change cell color intensity.

## Timeline stages

For frame `N`, the shared timeline uses these values:

| Value | Stage | Producer | Consumer |
| --- | --- | --- | --- |
| `4N + 1` | submit | main thread, host signal | compute and graphics workers |
| `4N + 2` | draw | compute queue signal | graphics queue wait |
| `4N + 3` | present | graphics queue signal | main thread, host wait |
| `4N + 4` | end of frame | main thread, host signal | next frame |

The main thread acquires a swapchain image and publishes an immutable frame
ticket before signaling `submit`. Separate compute and graphics workers record
and submit their commands. On separate queues, graphics may submit its wait for
`draw(N)` before compute submits the matching signal. Timeline semaphores permit
this ordering. If both workers use the same queue, graphics waits for `draw(N)`
on the host before submitting, avoiding a wait ahead of its own signal.

The graphics worker uses the presenter's borrowed image-ready semaphore,
render-finished semaphore, and fence. The main thread host-waits for
`present(N)`, presents through the presenter, then signals the end-of-frame
value. Workers are joined before the device is idled and resources are released.
The final counter value is printed on exit.

See [third-party notices](../THIRD_PARTY.md) for the shader license and source.
