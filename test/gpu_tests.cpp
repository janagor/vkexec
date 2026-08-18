#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/buffer.hpp>
#include <vkexec/bulk.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/config.hpp>
#include <vkexec/pass.hpp>
#include <vkexec_edsl/control.hpp>
#include <vkexec_edsl/push_constant.hpp>
#include <vkexec_edsl/types.hpp>

#include <boost/describe/class.hpp>

#include <stdexec/execution.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
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

auto skip_if_no_vulkan(std::exception const &error) -> void
{ SKIP(std::string("Vulkan unavailable: ") + error.what()); }

}// namespace

TEST_CASE("headless bulk compute updates buffers", "[vkexec][gpu]")
{
  constexpr std::size_t k_count = 128;
  constexpr float k_initial_velocity = 1.5F;
  constexpr float k_timestep = 0.016F;
  constexpr float k_damping = 0.99F;
  constexpr float k_epsilon = 1.0E-4F;

  std::optional<vkexec::context> ctx;
  VKEXEC_TRY
  {
    ctx.emplace();
  }
  VKEXEC_CATCH(std::exception const &error)
  {
    skip_if_no_vulkan(error);
  }
  vkexec::buffer<float> positions(*ctx, k_count, 0.0F);
  vkexec::buffer<float> velocities(*ctx, k_count, k_initial_velocity);

  sim_params const params{ .dt = k_timestep, .damping = k_damping };
  auto pipeline = ex::schedule(ctx->get_scheduler())
                  | vkexec::bulk(static_cast<std::uint32_t>(k_count),
                    params,
                    [&](edsl::Int idx, edsl::push_constant<sim_params> push) -> void {
                      edsl::Float position = positions[idx];
                      edsl::Float velocity = velocities[idx];
                      velocity = velocity * push.get<&sim_params::damping>();
                      position = position + (velocity * push.get<&sim_params::dt>());
                      positions[idx] = position;
                      velocities[idx] = velocity;
                    });
  ex::sync_wait(pipeline);

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

  std::optional<vkexec::context> ctx;
  VKEXEC_TRY
  {
    ctx.emplace();
  }
  VKEXEC_CATCH(std::exception const &error)
  {
    skip_if_no_vulkan(error);
  }
  vkexec::buffer<float> values(*ctx, k_count, k_initial);

  auto graph = ex::schedule(ctx->get_scheduler())
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
  ex::sync_wait(std::move(graph));

  float const expected = (k_initial + k_add) * k_scale;
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (std::size_t index = 0; index < k_count; ++index) {
    REQUIRE(std::fabs(values.data()[index] - expected) <= k_epsilon);
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

TEST_CASE("odd-even sort completes in one command buffer", "[vkexec][gpu]")
{
  constexpr std::size_t k_count = 32;
  constexpr unsigned k_rng_seed = 42;
  constexpr float k_value_max = 1000.0F;
  constexpr float k_epsilon = 1.0E-4F;

  std::optional<vkexec::context> ctx;
  VKEXEC_TRY
  {
    ctx.emplace();
  }
  VKEXEC_CATCH(std::exception const &error)
  {
    skip_if_no_vulkan(error);
  }
  vkexec::buffer<float> data(*ctx, k_count, 0.0F);

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

  auto graph = ex::schedule(ctx->get_scheduler()) | make_phase(0);
  for (std::size_t phase = 1; phase < k_count; ++phase) {
    graph = std::move(graph) | vkexec::barrier::compute_to_compute() | make_phase(phase);
  }
  ex::sync_wait(std::move(graph));

  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (std::size_t index = 0; index < k_count; ++index) {
    REQUIRE(std::fabs(data.data()[index] - expected.at(index)) <= k_epsilon);
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}
