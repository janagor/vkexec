#include <vkexec/barrier.hpp>
#include <vkexec/buffer.hpp>
#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec_edsl/push_constant.hpp>
#include <vkexec_edsl/types.hpp>

#include <boost/describe/class.hpp>

#include <stdexec/execution.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <print>
#include <utility>

namespace ex = stdexec;
namespace edsl = vkexec::edsl;

namespace {
constexpr std::size_t k_element_count = 1024;
constexpr float k_initial = 1.0F;
constexpr float k_add = 3.0F;
constexpr float k_scale = 2.0F;
constexpr float k_epsilon = 1.0E-4F;
}// namespace

struct pass_params
{
  float value;
};
// cppcheck-suppress unknownMacro
BOOST_DESCRIBE_STRUCT(pass_params, (), (value))

// NOLINTNEXTLINE(bugprone-exception-escape)
auto main() -> int
{
  try {
    vkexec::context ctx{ { .validation_layers = true } };
    vkexec::buffer<float> values(ctx, k_element_count, k_initial);

    auto graph = ex::schedule(ctx.get_scheduler())
                 | vkexec::compute_pass(static_cast<std::uint32_t>(k_element_count),
                   pass_params{ .value = k_add },
                   [&](edsl::Int idx, edsl::push_constant<pass_params> push) -> void {
                     values[idx] = values[idx] + push.get<&pass_params::value>();
                   })
                 | vkexec::barrier::compute_to_compute()
                 | vkexec::compute_pass(static_cast<std::uint32_t>(k_element_count),
                   pass_params{ .value = k_scale },
                   [&](edsl::Int idx, edsl::push_constant<pass_params> push) -> void {
                     values[idx] = values[idx] * push.get<&pass_params::value>();
                   });
    ex::sync_wait(std::move(graph));

    float const expected = (k_initial + k_add) * k_scale;
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    for (std::size_t index = 0; index < k_element_count; ++index) {
      if (std::fabs(values.data()[index] - expected) > k_epsilon) {
        std::println(stderr, "pass mismatch at {}: got {} expected {}", index, values.data()[index], expected);
        return 1;
      }
    }
    std::println("vkexec passes ok: N={} result={}", k_element_count, values.data()[0]);
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return 0;
  } catch (std::exception const &ex) {
    std::println(stderr, "vkexec passes example failed: {}", ex.what());
    return 1;
  }
}
