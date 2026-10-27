#include <vkexec/pass.hpp>
#include <vkexec_profiling/profiling.hpp>

#include <stdexec/execution.hpp>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

[[nodiscard]] auto make_profiled_graph(vkexec::context &ctx, vkexec::profiling::profiler profiler) -> auto
{
  return stdexec::schedule(ctx.get_scheduler())
         | vkexec::profiling::profiled(std::move(profiler), "installed",
             vkexec::custom_pass(vkexec::uses(), [](VkCommandBuffer /*cmd*/) -> void {}));
}

auto main() -> int
{
  auto output = std::ostringstream{};
  auto const events = std::vector<vkexec::profiling::profile_event>{
    { .name = "installed", .submission_id = 1, .duration_ns = 1'000.0,
      .status = vkexec::profiling::sample_status::valid, .start_tick = 10,
      .timestamp_valid_bits = 64, .timestamp_period_ns = 1.0 }
  };
  auto const exported = vkexec::profiling::write_chrome_trace(output, events);
  return exported && output.str().find("\"name\":\"installed\"") != std::string::npos ? 0 : 1;
}
