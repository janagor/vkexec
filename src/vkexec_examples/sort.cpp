#include <vkexec/vkexec.hpp>

#include <stdexec/execution.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace ex = stdexec;

struct SortParams {
  int offset;
  int n;
};
BOOST_DESCRIBE_STRUCT(SortParams, (), (offset, n))

auto main() -> int
{
  try {
    vkexec::context ctx;
    constexpr std::size_t N = 256;
    vkexec::buffer<float> data(ctx, N, 0.0F);

    std::mt19937 rng{ 42 };
    std::uniform_real_distribution<float> dist(0.0F, 1000.0F);
    std::vector<float> expected(N);
    for (std::size_t index = 0; index < N; ++index) {
      float const value = dist(rng);
      data.data()[index] = value;
      expected[index] = value;
    }
    std::sort(expected.begin(), expected.end());

    // Odd-even transposition sort: N phases, each comparing disjoint pairs.
    for (int phase = 0; phase < static_cast<int>(N); ++phase) {
      SortParams params{ phase % 2, static_cast<int>(N) };
      auto pass = ex::schedule(ctx.get_scheduler())
                  | vkexec::bulk(static_cast<std::uint32_t>(N / 2), params,
                      [&](vkexec::Int idx, vkexec::PushConstant<SortParams> pc) {
                        vkexec::Int left = vkexec::Int::constant(2) * idx + pc.get<&SortParams::offset>();
                        vkexec::Int right = left + vkexec::Int::constant(1);

                        vkexec::if_then(right < pc.get<&SortParams::n>(), [&] {
                          vkexec::Float left_value = data[left];
                          vkexec::Float right_value = data[right];
                          vkexec::Bool out_of_order = left_value > right_value;
                          data[left] = vkexec::select(out_of_order, right_value, left_value);
                          data[right] = vkexec::select(out_of_order, left_value, right_value);
                        });
                      });
      ex::sync_wait(std::move(pass));
    }

    for (std::size_t index = 0; index < N; ++index) {
      if (std::fabs(data.data()[index] - expected[index]) > 1e-4F) {
        std::fprintf(stderr,
          "sort mismatch at %zu: got %f expected %f\n",
          index,
          static_cast<double>(data.data()[index]),
          static_cast<double>(expected[index]));
        return 1;
      }
    }

    std::printf(
      "vkexec sort ok: N=%zu first=%f mid=%f last=%f\n",
      N,
      static_cast<double>(data.data()[0]),
      static_cast<double>(data.data()[N / 2]),
      static_cast<double>(data.data()[N - 1]));
    return 0;
  } catch (std::exception const &ex) {
    std::fprintf(stderr, "vkexec sort example failed: %s\n", ex.what());
    return 1;
  }
}
