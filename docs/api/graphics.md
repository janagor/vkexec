# Graphics

Graphics pipelines and presentation interfaces.

`record_draw` requires an active compatible render pass and leaves it open.
`record_draw_pass` and `graphics_pipeline::record_pass` begin and end a render pass;
the command buffer remains recording. The owner that began command-buffer
recording ends it: `presenter::end_frame` does this for normal frames, while
`presenter::finish_frame_recording` does it before caller-managed submission.

## Public headers

- [`<vkexec_graphics/graphics_pipeline_resources.hpp>`](https://github.com/janagor/vkexec/blob/main/include/vkexec_graphics/graphics_pipeline_resources.hpp)
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
