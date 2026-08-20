#include <vkexec/barrier.hpp>
#include <vkexec/buffer.hpp>
#include <vkexec/context.hpp>
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
#include <print>
#include <random>
#include <utility>
#include <vector>

namespace ex = stdexec;
namespace edsl = vkexec::edsl;

namespace {
constexpr std::size_t k_element_count = 256;
constexpr unsigned k_rng_seed = 42;
constexpr float k_value_max = 1000.0F;
constexpr float k_epsilon = 1.0E-4F;
}// namespace

struct sort_params
{
  int offset;
  int n;
};
// cppcheck-suppress unknownMacro
BOOST_DESCRIBE_STRUCT(sort_params, (), (offset, n))

// NOLINTNEXTLINE(bugprone-exception-escape)
auto main() -> int
{
  try {
    vkexec::context ctx{ { .validation_layers = true } };
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    auto [data] = ex::sync_wait(vkexec::buffer<float>::allocate(ctx, k_element_count, 0.0F)).value();

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

    auto make_phase = [&](std::size_t phase) -> auto {
      sort_params const params{ .offset = static_cast<int>(phase % 2), .n = static_cast<int>(k_element_count) };
      return vkexec::compute_pass(static_cast<std::uint32_t>(k_element_count / 2),
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

    // Odd-even transposition sort: N phases in one command buffer, barrier between each.
    auto graph = ex::schedule(ctx.get_scheduler()) | make_phase(0);
    for (std::size_t phase = 1; phase < k_element_count; ++phase) {
      graph = std::move(graph) | vkexec::barrier::compute_to_compute() | make_phase(phase);
    }
    ex::sync_wait(std::move(graph));

    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    for (std::size_t index = 0; index < k_element_count; ++index) {
      if (std::fabs(data.data()[index] - expected.at(index)) > k_epsilon) {
        std::println(stderr, "sort mismatch at {}: got {} expected {}", index, data.data()[index], expected.at(index));
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
