# Presentation

The graphics module offers presenter, swapchain, pipeline, and draw helpers. The application chooses the window system and supplies a surface factory. This keeps GLFW or another window library outside vkexec's public execution API.

## Supply a surface factory

The compiled triangle example uses a GLFW adapter in `examples/glfw_presenter.hpp`. It gathers required instance extensions, creates a Vulkan surface from the GLFW window, and passes that callback to `factory::make_presenter`. The presenter configuration also accepts Vulkan requirements and a depth-attachment factory.

```{literalinclude} ../../examples/glfw_presenter.hpp
:language: cpp
:start-at: created.presenter_ = std::make_unique<owned::presenter>
:end-at: return created;
```

The window and surface must remain valid while the presenter uses them. Recreate swapchain-dependent resources when the window size or surface state changes; the presenter and adapter own the corresponding cleanup path.

## Draw through the scheduler

The [triangle program](https://github.com/janagor/vkexec/blob/main/examples/triangle.cpp) creates a graphics pipeline and composes a draw sender for each frame. It explicitly waits per frame in this example; the sender model also supports asynchronous composition when the application manages frame overlap and lifetimes.

```{literalinclude} ../../examples/triangle.cpp
:language: cpp
:start-at: while (!win.should_close()) {
:end-at: win.wait_idle();
```

The optional timeline-semaphore module provides additional frame coordination helpers. See [Graphics API](../api/graphics.md) for the public presenter and pipeline types.
