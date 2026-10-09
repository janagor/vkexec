# Graphics

Graphics pipelines and presentation interfaces.

`record_draw` requires an active compatible render pass and leaves it open.
`record_draw_pass` and `graphics_pipeline::record_pass` begin and end a render pass;
the command buffer remains recording. The owner that began command-buffer
recording ends it: `presenter::end_frame` does this for normal frames, while
`presenter::finish_frame_recording` does it before caller-managed submission.

`graphics_pass(target, bind, draw)` adds a draw to a pass graph. The target names
the underlying color and depth images as well as the render pass and framebuffer.
Each `graphics_attachment` supplies its graph-entry layout, the render pass's
`initialLayout` and `finalLayout`, and access including any `LOAD`, depth-test,
or blending reads. When the render pass initial layout is `UNDEFINED`, the
render pass owns the transition into the color or depth attachment layout.
The render-pass initial layout must be set explicitly, including when it is
`UNDEFINED`; the final layout must be a defined Vulkan layout.
The attachment layout during the pass is inferred from its color or depth role.
The pipeline config comes from `bind`, and descriptor accesses come from the
resource table retained by `owned::graphics_pipeline::bind()` or passed to
`bind_graphics(pipe, set, table)`. Storage descriptors are conservatively
treated as read/write across graphics shader stages. A graph pass rejects a
binding whose descriptor metadata is incomplete.

## Public headers

- [`<vkexec_graphics/graphics_pipeline_resources.hpp>`](https://github.com/janagor/vkexec/blob/main/include/vkexec_graphics/graphics_pipeline_resources.hpp)
- [`<vkexec_graphics/pass.hpp>`](https://github.com/janagor/vkexec/blob/main/include/vkexec_graphics/pass.hpp)
- [`<vkexec_graphics/presenter.hpp>`](https://github.com/janagor/vkexec/blob/main/include/vkexec_graphics/presenter.hpp)

## Representative declarations

```{doxygenstruct} vkexec::handles::graphics_pipeline
:project: vkexec
:outline:
```

```{doxygenclass} vkexec::owned::graphics_pipeline
:project: vkexec
:outline:
```

```{doxygenclass} vkexec::owned::presenter
:project: vkexec
:outline:
```
