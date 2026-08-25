#include <vkexec/buffer.hpp>
#include <vkexec/bulk.hpp>
#include <vkexec/context.hpp>
#include <vkexec_edsl/push_constant.hpp>
#include <vkexec_edsl/types.hpp>

#include <boost/describe/class.hpp>

#include <stdexec/execution.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <print>

namespace ex = stdexec;
namespace edsl = vkexec::edsl;

namespace {
constexpr std::size_t k_element_count = 10000;
constexpr float k_initial_velocity = 1.5F;
constexpr float k_timestep = 0.016F;
constexpr float k_damping = 0.99F;
constexpr float k_epsilon = 1.0E-4F;
}// namespace

struct sim_params
{
  float dt;
  float damping;
};
// cppcheck-suppress unknownMacro
BOOST_DESCRIBE_STRUCT(sim_params, (), (dt, damping))

// NOLINTNEXTLINE(bugprone-exception-escape)
auto main() -> int
{
  try {
    vkexec::context ctx{ { .validation_layers = true } };
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    auto [positions, velocities] =
      ex::sync_wait(ex::when_all(vkexec::buffer<float>::allocate(ctx, k_element_count, 0.0F),
                      vkexec::buffer<float>::allocate(ctx, k_element_count, k_initial_velocity)))
        .value();
    // NOLINTEND(bugprone-unchecked-optional-access)

    sim_params const params{ .dt = k_timestep, .damping = k_damping };

    auto pipeline = ex::schedule(ctx.get_scheduler())
                    | vkexec::bulk(static_cast<std::uint32_t>(k_element_count),
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
    for (std::size_t const index : { std::size_t{ 0 }, k_element_count / 2, k_element_count - 1 }) {
      if (std::fabs(velocities.data()[index] - expected_v) > k_epsilon
          || std::fabs(positions.data()[index] - expected_p) > k_epsilon) {
        std::println(stderr,
          "mismatch at {}: p={} v={} (expected p={} v={})",
          index,
          positions.data()[index],
          velocities.data()[index],
          expected_p,
          expected_v);
        return 1;
      }
    }

    std::println("vkexec sim ok: p[0]={} v[0]={}", positions.data()[0], velocities.data()[0]);
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return 0;
  } catch (std::exception const &ex) {
    std::println(stderr, "vkexec example failed: {}", ex.what());
    return 1;
  }
}
