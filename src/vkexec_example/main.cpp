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

int main()
{
  try {
    vkexec::context ctx;
    constexpr std::size_t N = 10000;
    vkexec::buffer<float> positions(ctx, N, 0.0f);
    vkexec::buffer<float> velocities(ctx, N, 1.5f);

    SimParams params{ 0.016f, 0.99f };

    auto pipeline = ex::schedule(ctx.get_scheduler())
                    | vkexec::bulk(static_cast<std::uint32_t>(N), params,
                        [&](vkexec::Int idx, vkexec::PushConstant<SimParams> pc) {
                          vkexec::Float p = positions[idx];
                          vkexec::Float v = velocities[idx];

                          v = v * pc.get<&SimParams::damping>();
                          p = p + (v * pc.get<&SimParams::dt>());

                          positions[idx] = p;
                          velocities[idx] = v;
                        });

    ex::sync_wait(std::move(pipeline));

    const float expected_v = 1.5f * 0.99f;
    const float expected_p = expected_v * 0.016f;
    for (std::size_t i : { std::size_t{ 0 }, N / 2, N - 1 }) {
      if (std::fabs(velocities.data()[i] - expected_v) > 1e-4f
          || std::fabs(positions.data()[i] - expected_p) > 1e-4f) {
        std::fprintf(stderr, "mismatch at %zu: p=%f v=%f (expected p=%f v=%f)\n", i,
          static_cast<double>(positions.data()[i]), static_cast<double>(velocities.data()[i]),
          static_cast<double>(expected_p), static_cast<double>(expected_v));
        return 1;
      }
    }

    std::printf("vkexec sim ok: p[0]=%f v[0]=%f\n", static_cast<double>(positions.data()[0]),
      static_cast<double>(velocities.data()[0]));
    return 0;
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "vkexec example failed: %s\n", ex.what());
    return 1;
  }
}
