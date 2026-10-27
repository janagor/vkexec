#ifndef VKEXEC_PROFILING_DETAIL_TIMESTAMP_MATH_HPP
#define VKEXEC_PROFILING_DETAIL_TIMESTAMP_MATH_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace vkexec::profiling::detail {

[[nodiscard]] inline auto duration_ns(std::uint64_t start, std::uint64_t end, std::uint32_t valid_bits, float period)
  -> std::optional<double>
{
  constexpr std::uint32_t k_counter_bits = std::numeric_limits<std::uint64_t>::digits;
  if (valid_bits == 0 || valid_bits > k_counter_bits || !std::isfinite(period) || period <= 0.0F) {
    return std::nullopt;
  }
  auto delta = end - start;
  if (valid_bits < k_counter_bits) { delta &= (std::uint64_t{ 1 } << valid_bits) - 1; }
  auto const duration = static_cast<double>(delta) * static_cast<double>(period);
  if (!std::isfinite(duration)) { return std::nullopt; }
  return duration;
}

}// namespace vkexec::profiling::detail

#endif// VKEXEC_PROFILING_DETAIL_TIMESTAMP_MATH_HPP
