#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/buffer.hpp>
#include <vkexec/bulk.hpp>
#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/submit.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_edsl/control.hpp>
#include <vkexec_edsl/push_constant.hpp>
#include <vkexec_edsl/types.hpp>

#include <boost/describe/class.hpp>

#include <stdexec/execution.hpp>
#include <stdexec/stop_token.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace ex = stdexec;
namespace edsl = vkexec::edsl;

namespace {

struct sim_params
{
  float dt;
  float damping;
};
// cppcheck-suppress unknownMacro
BOOST_DESCRIBE_STRUCT(sim_params, (), (dt, damping))

struct pass_params
{
  float value;
};
// cppcheck-suppress unknownMacro
BOOST_DESCRIBE_STRUCT(pass_params, (), (value))

struct sort_params
{
  int offset;
  int n;
};
// cppcheck-suppress unknownMacro
BOOST_DESCRIBE_STRUCT(sort_params, (), (offset, n))

auto skip_if_no_vulkan(vkexec::error const &err) -> void
{ SKIP(std::string("Vulkan unavailable: ") + std::string(err.message())); }

}// namespace

TEST_CASE("headless bulk compute updates buffers", "[vkexec][gpu]")
{
  constexpr std::size_t k_count = 128;
  constexpr float k_initial_velocity = 1.5F;
  constexpr float k_timestep = 0.016F;
  constexpr float k_damping = 0.99F;
  constexpr float k_epsilon = 1.0E-4F;

  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto positions_result = vkexec::buffer<float>::create_sync(ctx, k_count, 0.0F);
  REQUIRE(positions_result.has_value());
  auto velocities_result = vkexec::buffer<float>::create_sync(ctx, k_count, k_initial_velocity);
  REQUIRE(velocities_result.has_value());
  auto &positions = *positions_result;
  auto &velocities = *velocities_result;

  sim_params const params{ .dt = k_timestep, .damping = k_damping };
  auto pipeline =
    ex::schedule(ctx.get_scheduler())
    | vkexec::bulk(
      static_cast<std::uint32_t>(k_count), params, [&](edsl::Int idx, edsl::push_constant<sim_params> push) -> void {
        edsl::Float position = positions[idx];
        edsl::Float velocity = velocities[idx];
        velocity = velocity * push.get<&sim_params::damping>();
        position = position + (velocity * push.get<&sim_params::dt>());
        positions[idx] = position;
        velocities[idx] = velocity;
      });
  auto waited = vkexec::sync_wait(pipeline);
  REQUIRE(waited.has_value());
  REQUIRE(waited->has_value());

  float const expected_v = k_initial_velocity * k_damping;
  float const expected_p = expected_v * k_timestep;
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  REQUIRE(std::fabs(positions.data()[0] - expected_p) <= k_epsilon);
  REQUIRE(std::fabs(velocities.data()[0] - expected_v) <= k_epsilon);
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

TEST_CASE("chained compute passes reuse descriptor sets safely", "[vkexec][gpu]")
{
  constexpr std::size_t k_count = 64;
  constexpr float k_initial = 1.0F;
  constexpr float k_add = 3.0F;
  constexpr float k_scale = 2.0F;
  constexpr float k_epsilon = 1.0E-4F;

  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto values_result = vkexec::buffer<float>::create_sync(ctx, k_count, k_initial);
  REQUIRE(values_result.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  auto &values = *values_result;

  auto graph = ex::schedule(ctx.get_scheduler())
               | vkexec::compute_pass(static_cast<std::uint32_t>(k_count),
                 pass_params{ .value = k_add },
                 [&](edsl::Int idx, edsl::push_constant<pass_params> push) -> void {
                   values[idx] = values[idx] + push.get<&pass_params::value>();
                 })
               | vkexec::barrier::compute_to_compute()
               | vkexec::compute_pass(static_cast<std::uint32_t>(k_count),
                 pass_params{ .value = k_scale },
                 [&](edsl::Int idx, edsl::push_constant<pass_params> push) -> void {
                   values[idx] = values[idx] * push.get<&pass_params::value>();
                 });
  auto waited = vkexec::sync_wait(std::move(graph));
  REQUIRE(waited.has_value());
  REQUIRE(waited->has_value());

  float const expected = (k_initial + k_add) * k_scale;
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (std::size_t index = 0; index < k_count; ++index) {
    REQUIRE(std::fabs(values.data()[index] - expected) <= k_epsilon);
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

TEST_CASE("chained compute_pass graph completes asynchronously", "[vkexec][gpu]")
{
  constexpr std::size_t k_count = 64;
  constexpr float k_initial = 1.0F;
  constexpr float k_add = 3.0F;
  constexpr float k_scale = 2.0F;
  constexpr float k_epsilon = 1.0E-4F;

  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto values_result = vkexec::buffer<float>::create_sync(ctx, k_count, k_initial);
  REQUIRE(values_result.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  auto &values = *values_result;

  auto graph = ex::schedule(ctx.get_scheduler())
               | vkexec::compute_pass(static_cast<std::uint32_t>(k_count),
                 pass_params{ .value = k_add },
                 [&](edsl::Int idx, edsl::push_constant<pass_params> push) -> void {
                   values[idx] = values[idx] + push.get<&pass_params::value>();
                 })
               | vkexec::barrier::compute_to_compute()
               | vkexec::compute_pass(static_cast<std::uint32_t>(k_count),
                 pass_params{ .value = k_scale },
                 [&](edsl::Int idx, edsl::push_constant<pass_params> push) -> void {
                   values[idx] = values[idx] * push.get<&pass_params::value>();
                 })
               | vkexec::submit;
  auto waited = vkexec::sync_wait(graph);
  REQUIRE(waited.has_value());
  REQUIRE(waited->has_value());

  float const expected = (k_initial + k_add) * k_scale;
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (std::size_t index = 0; index < k_count; ++index) {
    REQUIRE(std::fabs(values.data()[index] - expected) <= k_epsilon);
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

TEST_CASE("pass graph submit completes with set_stopped when stop is already requested", "[vkexec][pass]")
{
  vkexec::scheduler const sched{ nullptr };
  ex::inplace_stop_source source;
  source.request_stop();

  auto sender =
    ex::schedule(sched)
    | vkexec::compute_pass(
      1U, pass_params{ .value = 1.0F }, [](edsl::Int /*idx*/, edsl::push_constant<pass_params> /*push*/) -> void {})
    | vkexec::submit;

  auto const waited =
    // NOLINTNEXTLINE(misc-include-cleaner)
    vkexec::sync_wait(ex::write_env(sender, ex::prop{ ex::get_stop_token, source.get_token() }));
  REQUIRE(waited.has_value());
  REQUIRE_FALSE(waited->has_value());
}

TEST_CASE("odd-even sort completes in one command buffer", "[vkexec][gpu]")
{
  constexpr std::size_t k_count = 32;
  constexpr unsigned k_rng_seed = 42;
  constexpr float k_value_max = 1000.0F;
  constexpr float k_epsilon = 1.0E-4F;

  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto data_result = vkexec::buffer<float>::create_sync(ctx, k_count, 0.0F);
  REQUIRE(data_result.has_value());
  auto &data = *data_result;

  // NOLINTNEXTLINE(bugprone-random-generator-seed,cert-msc32-c,cert-msc51-cpp)
  std::mt19937 rng{ k_rng_seed };
  std::uniform_real_distribution<float> dist(0.0F, k_value_max);
  std::vector<float> expected(k_count);
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (std::size_t index = 0; index < k_count; ++index) {
    float const value = dist(rng);
    data.data()[index] = value;
    expected.at(index) = value;
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::ranges::sort(expected);

  auto make_phase = [&](std::size_t phase) -> auto {
    sort_params const params{ .offset = static_cast<int>(phase % 2), .n = static_cast<int>(k_count) };
    return vkexec::compute_pass(static_cast<std::uint32_t>(k_count / 2),
      params,
      [&](edsl::Int idx, edsl::push_constant<sort_params> push) -> void {
        edsl::Int left = edsl::Int::constant(2) * idx + push.get<&sort_params::offset>();
        edsl::Int right = left + edsl::Int::constant(1);
        edsl::if_then(right < push.get<&sort_params::n>(), [&]() -> void {
          edsl::Float const left_value = data[left];
          edsl::Float const right_value = data[right];
          edsl::Bool const out_of_order = left_value > right_value;
          data[left] = edsl::select(out_of_order, right_value, left_value);
          data[right] = edsl::select(out_of_order, left_value, right_value);
        });
      });
  };

  auto graph = ex::schedule(ctx.get_scheduler()) | make_phase(0);
  for (std::size_t phase = 1; phase < k_count; ++phase) {
    graph = std::move(graph) | vkexec::barrier::compute_to_compute() | make_phase(phase);
  }
  auto waited = vkexec::sync_wait(std::move(graph));
  REQUIRE(waited.has_value());
  REQUIRE(waited->has_value());

  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (std::size_t index = 0; index < k_count; ++index) {
    REQUIRE(std::fabs(data.data()[index] - expected.at(index)) <= k_epsilon);
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}
