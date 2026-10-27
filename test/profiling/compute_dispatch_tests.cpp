#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/compute_pipeline.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec_profiling/profile_event.hpp>
#include <vkexec_profiling/profiled.hpp>
#include <vkexec_profiling/profiler.hpp>

#include <stdexec/execution.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

namespace {

constexpr std::string_view k_compute_glsl = R"(
#version 450
layout(local_size_x = 1) in;
void main() {}
)";
constexpr std::uint32_t k_local_size = 1;
constexpr std::size_t k_push_constant_size = 0;

}// namespace

TEST_CASE("profiled compute dispatch reports a valid GPU sample", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx);
  REQUIRE(profiler);
  if (!profiler->supported(ctx->compute_queue_ref())) { SKIP("compute queue has no timestamp support"); }
  auto pipeline = vkexec::test::sync_wait_value(vkexec::factory::make_compute_pipeline(*ctx,
    k_compute_glsl,
    vkexec::layout_desc{ .binding_kinds = {}, .binding_slots = {}, .bindings = {},
                         .push_constant_size = k_push_constant_size, .specialization = {},
                         .local_size = { k_local_size, k_local_size, k_local_size } },
    "profiled.comp"));
  auto bind = vkexec::bind_compute(pipeline.resources()).with_resources({}, {});
  auto graph = stdexec::schedule(ctx->get_scheduler())
               | vkexec::profiling::profiled(*profiler, "dispatch",
                   vkexec::compute_pass(std::move(bind), vkexec::dispatch{}));
  auto outcome = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(outcome));
  auto events = profiler->take_completed();
  constexpr std::size_t k_sample_count = 1;
  REQUIRE(events.size() == k_sample_count);
  REQUIRE(events.front().status == vkexec::profiling::sample_status::valid);
}
