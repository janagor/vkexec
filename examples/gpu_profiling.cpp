#include "sync_wait_helpers.hpp"

#include <vkexec/buffer.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec_profiling/profiling.hpp>

#include <stdexec/execution.hpp>

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr std::uint32_t k_element_count = 1'024;
constexpr std::uint32_t k_local_size = 64;
constexpr float k_initial_value = 1.0F;
constexpr float k_first_add = 2.0F;
constexpr float k_second_add = 3.0F;
constexpr std::string_view k_shader = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer Values { float data[]; } values;
layout(push_constant) uniform Push { float addend; } pc;
void main() { values.data[gl_GlobalInvocationID.x] += pc.addend; }
)";

struct parameters { float addend; };

auto run() -> int
{
  auto ctx = vkexec::examples::sync_wait_value(vkexec::factory::make_context());
  auto allocator = vkexec::examples::make_vma_allocator(*ctx);
  auto values = vkexec::examples::sync_wait_value(
    vkexec::factory::make_buffer(allocator, k_element_count, k_initial_value));
  auto pipeline = vkexec::examples::sync_wait_value(vkexec::factory::make_compute_pipeline(*ctx,
    k_shader,
    vkexec::layout_desc{
      .binding_kinds = {},
      .binding_slots = {},
      .bindings = { vkexec::buffer_access::readwrite },
      .push_constant_size = sizeof(parameters),
      .specialization = {},
      .local_size = { k_local_size, 1, 1 },
    },
    "profiling.comp"));
  std::array<vkexec::storage_binding, 1> const buffers{ vkexec::storage_binding{
    .buffer = values.vk_buffer(), .byte_size = values.size() * sizeof(float) } };
  auto bound = vkexec::examples::sync_wait_value(vkexec::bind_storage_sender(pipeline, buffers));
  auto profiler = vkexec::profiling::profiler::create(*ctx);
  if (!profiler) {
    std::cerr << "profiler initialization failed\n";
    return 1;
  }

  auto graph = stdexec::schedule(ctx->get_scheduler())
    | vkexec::profiling::profiled(*profiler, "first-add",
        vkexec::compute_pass(bound, parameters{ k_first_add }, k_element_count))
    | vkexec::profiling::profiled(*profiler, "second-add",
        vkexec::compute_pass(bound, parameters{ k_second_add }, k_element_count));
  vkexec::examples::sync_wait_graph(graph);
  constexpr std::array<std::uint32_t, 3> k_check_indices{ 0, k_element_count / 2, k_element_count - 1 };
  constexpr auto k_expected_value = k_initial_value + k_first_add + k_second_add;
  for (auto const index : k_check_indices) {
    if (values.data()[index] != k_expected_value) {
      vkexec::examples::fail_check("profiled compute result mismatch");
    }
  }

  auto const events = profiler->take_completed();
  constexpr std::size_t k_expected_events = 2;
  if (events.size() != k_expected_events) { vkexec::examples::fail_check("missing profiling events"); }
  for (auto const &event : events) {
    if (event.status != vkexec::profiling::sample_status::valid) {
      vkexec::examples::fail_check("GPU timestamp sample is unavailable");
    }
    std::cout << event.name << ": " << event.duration_ns / 1'000'000.0 << " ms\n";
  }
  auto submissions = std::map<std::uint64_t, std::vector<vkexec::profiling::profile_event>>{};
  for (auto const &event : events) { submissions[event.submission_id].push_back(event); }
  for (auto const &[submission_id, samples] : submissions) {
    auto const filename = "gpu-trace-" + std::to_string(submission_id) + ".json";
    auto output = std::ofstream(filename);
    auto exported = vkexec::profiling::write_chrome_trace(output, samples);
    if (!exported) {
      std::cerr << "trace export failed: " << exported.error().message() << '\n';
      return 1;
    }
    std::cout << "wrote " << filename << '\n';
  }
  return 0;
}
}// namespace

auto main() -> int { return vkexec::examples::run_example(run); }
