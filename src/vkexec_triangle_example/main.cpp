#include <vkexec/vkexec.hpp>

#include <stdexec/execution.hpp>

#include <cstdio>
#include <cstdlib>

namespace ex = stdexec;

int main()
{
  try {
    vkexec::window win({ .width = 800, .height = 600, .title = "vkexec triangle" });

    // Vertex + fragment shaders traced from C++ (AST → GLSL → SPIR-V).
    vkexec::graphics_pipeline pipeline(win.ctx(),
      win.render_pass(),
      [](vlk::Int vid, vlk::VertexWriter out) {
        const vlk::Float2 pos = vlk::select(vid == vlk::Int::constant(0),
          vlk::vec2(0.0, -0.5),
          vlk::select(vid == vlk::Int::constant(1), vlk::vec2(0.5, 0.5), vlk::vec2(-0.5, 0.5)));
        const vlk::Float3 col = vlk::select(vid == vlk::Int::constant(0),
          vlk::vec3(1.0, 0.2, 0.2),
          vlk::select(vid == vlk::Int::constant(1), vlk::vec3(0.2, 1.0, 0.2), vlk::vec3(0.2, 0.4, 1.0)));
        out.position(pos);
        out.color(col);
      },
      [](vlk::FragmentReader in, vlk::FragmentWriter out) { out.color(vlk::vec4(in.color(), 1.0)); });

    std::printf("vkexec traced triangle (stdexec frame pipeline) — close the window to exit\n");
    while (!win.should_close()) {
      win.poll_events();
      // Same shape as compute: schedule | algorithm | sync_wait
      ex::sync_wait(ex::schedule(win.ctx().get_scheduler()) | vkexec::draw(win, pipeline, 3));
    }
    win.wait_idle();
    return 0;
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "vkexec triangle example failed: %s\n", ex.what());
    return 1;
  }
}
