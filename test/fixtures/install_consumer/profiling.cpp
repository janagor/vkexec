#include <vkexec/pass.hpp>
#include <vkexec_profiling/profiling.hpp>

#include <stdexec/execution.hpp>

#include <utility>

[[nodiscard]] auto make_profiled_graph(vkexec::context &ctx, vkexec::profiling::profiler profiler) -> auto
{
  return stdexec::schedule(ctx.get_scheduler())
         | vkexec::profiling::profiled(std::move(profiler), "installed",
             vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {}));
}

auto main() -> int { return 0; }
