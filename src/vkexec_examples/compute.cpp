#include <vkexec/vkexec.hpp>

#include <stdexec/execution.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace ex = stdexec;

struct SimParams {
  float dt;
  float damping;
};
BOOST_DESCRIBE_STRUCT(SimParams, (), (dt, damping))

auto main() -> int
{
  try {
    vkexec::context ctx;
    constexpr std::size_t N = 10000;
    vkexec::buffer<float> positions(ctx, N, 0.0F);
    vkexec::buffer<float> velocities(ctx, N, 1.5F);

    SimParams params{ 0.016F, 0.99F };

    auto pipeline = ex::schedule(ctx.get_scheduler())
                    | vkexec::bulk(static_cast<std::uint32_t>(N), params,
                        [&](vkexec::Int idx, vkexec::PushConstant<SimParams> pc) {
                          vkexec::Float position = positions[idx];
                          vkexec::Float velocity = velocities[idx];

                          velocity = velocity * pc.get<&SimParams::damping>();
                          position = position + (velocity * pc.get<&SimParams::dt>());

                          positions[idx] = position;
                          velocities[idx] = velocity;
                        });

    ex::sync_wait(std::move(pipeline));

    float const expected_v = 1.5F * 0.99F;
    float const expected_p = expected_v * 0.016F;
    for (std::size_t index : { std::size_t{ 0 }, N / 2, N - 1 }) {
      if (std::fabs(velocities.data()[index] - expected_v) > 1e-4F
          || std::fabs(positions.data()[index] - expected_p) > 1e-4F) {
        std::fprintf(stderr,
          "mismatch at %zu: p=%f v=%f (expected p=%f v=%f)\n",
          index,
          static_cast<double>(positions.data()[index]),
          static_cast<double>(velocities.data()[index]),
          static_cast<double>(expected_p),
          static_cast<double>(expected_v));
        return 1;
      }
    }

    std::printf(
      "vkexec sim ok: p[0]=%f v[0]=%f\n", static_cast<double>(positions.data()[0]), static_cast<double>(velocities.data()[0]));
    return 0;
  } catch (std::exception const &ex) {
    std::fprintf(stderr, "vkexec example failed: %s\n", ex.what());
    return 1;
  }
}
