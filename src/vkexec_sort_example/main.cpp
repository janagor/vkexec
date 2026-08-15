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

VLK_PUSH_CONSTANT(SortParams, int, offset, int, n);

int main()
{
  try {
    vkexec::context ctx;
    constexpr std::size_t N = 256;
    vkexec::buffer<float> data(ctx, N, 0.0f);

    std::mt19937 rng{ 42 };
    std::uniform_real_distribution<float> dist(0.0f, 1000.0f);
    std::vector<float> expected(N);
    for (std::size_t i = 0; i < N; ++i) {
      const float v = dist(rng);
      data.data()[i] = v;
      expected[i] = v;
    }
    std::sort(expected.begin(), expected.end());

    // Odd-even transposition sort: N phases, each comparing disjoint pairs.
    for (int phase = 0; phase < static_cast<int>(N); ++phase) {
      SortParams params{ phase % 2, static_cast<int>(N) };
      auto pass = ex::schedule(ctx.get_scheduler())
                  | vkexec::bulk(static_cast<std::uint32_t>(N / 2), params,
                      [&](vlk::Int idx, vlk::PushConstant<SortParams> pc) {
                        vlk::Int left = vlk::Int::constant(2) * idx + pc.offset;
                        vlk::Int right = left + vlk::Int::constant(1);

                        vlk::if_then(right < pc.n, [&] {
                          vlk::Float a = data[left];
                          vlk::Float b = data[right];
                          vlk::Bool out_of_order = a > b;
                          data[left] = vlk::select(out_of_order, b, a);
                          data[right] = vlk::select(out_of_order, a, b);
                        });
                      });
      ex::sync_wait(std::move(pass));
    }

    for (std::size_t i = 0; i < N; ++i) {
      if (std::fabs(data.data()[i] - expected[i]) > 1e-4f) {
        std::fprintf(stderr, "sort mismatch at %zu: got %f expected %f\n", i, static_cast<double>(data.data()[i]),
          static_cast<double>(expected[i]));
        return 1;
      }
    }

    std::printf("vkexec sort ok: N=%zu first=%f mid=%f last=%f\n", N, static_cast<double>(data.data()[0]),
      static_cast<double>(data.data()[N / 2]), static_cast<double>(data.data()[N - 1]));
    return 0;
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "vkexec sort example failed: %s\n", ex.what());
    return 1;
  }
}
