#include <vkexec/bulk.hpp>
#include <vkexec/buffer.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/control.hpp>
#include <vkexec/detail/push_constant.hpp>
#include <vkexec/detail/types.hpp>

#include <boost/describe/class.hpp>

#include <stdexec/execution.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <print>
#include <random>
#include <vector>

namespace ex = stdexec;

namespace {
constexpr std::size_t k_element_count = 256;
constexpr unsigned k_rng_seed = 42;
constexpr float k_value_max = 1000.0F;
constexpr float k_epsilon = 1.0E-4F;
} // namespace

struct sort_params {
  int offset;
  int n;
};
BOOST_DESCRIBE_STRUCT(sort_params, (), (offset, n))

// NOLINTNEXTLINE(bugprone-exception-escape)
auto main() -> int
{
  try {
    vkexec::context ctx;
    vkexec::buffer<float> data(ctx, k_element_count, 0.0F);

    // NOLINTNEXTLINE(bugprone-random-generator-seed,cert-msc32-c,cert-msc51-cpp)
    std::mt19937 rng{ k_rng_seed };
    std::uniform_real_distribution<float> dist(0.0F, k_value_max);
    std::vector<float> expected(k_element_count);
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    for (std::size_t index = 0; index < k_element_count; ++index) {
      float const value = dist(rng);
      data.data()[index] = value;
      expected.at(index) = value;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::ranges::sort(expected);

    // Odd-even transposition sort: N phases, each comparing disjoint pairs.
    for (std::size_t phase = 0; phase < k_element_count; ++phase) {
      sort_params const params{ .offset = static_cast<int>(phase % 2), .n = static_cast<int>(k_element_count) };
      auto pass = ex::schedule(ctx.get_scheduler())
                  | vkexec::bulk(static_cast<std::uint32_t>(k_element_count / 2), params,
                      [&](vkexec::Int idx, vkexec::push_constant<sort_params> push) -> void {
                        vkexec::Int left = vkexec::Int::constant(2) * idx + push.get<&sort_params::offset>();
                        vkexec::Int right = left + vkexec::Int::constant(1);

                        vkexec::if_then(right < push.get<&sort_params::n>(), [&]() -> void {
                          // NOLINTBEGIN(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                          vkexec::Float const left_value = data[left];
                          vkexec::Float const right_value = data[right];
                          vkexec::Bool const out_of_order = left_value > right_value;
                          data[left] = vkexec::select(out_of_order, right_value, left_value);
                          data[right] = vkexec::select(out_of_order, left_value, right_value);
                          // NOLINTEND(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
                        });
                      });
      ex::sync_wait(pass);
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    for (std::size_t index = 0; index < k_element_count; ++index) {
      if (std::fabs(data.data()[index] - expected.at(index)) > k_epsilon) {
        std::println(stderr,
          "sort mismatch at {}: got {} expected {}",
          index,
          data.data()[index],
          expected.at(index));
        return 1;
      }
    }

    std::println("vkexec sort ok: N={} first={} mid={} last={}",
      k_element_count,
      data.data()[0],
      data.data()[k_element_count / 2],
      data.data()[k_element_count - 1]);
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return 0;
  } catch (std::exception const &ex) {
    std::println(stderr, "vkexec sort example failed: {}", ex.what());
    return 1;
  }
}
