#include <catch2/catch_test_macros.hpp>

#include <vkexec_profiling/detail/timestamp_math.hpp>

#include <cstdint>
#include <limits>

TEST_CASE("timestamp subtraction respects the queue family's valid bits", "[vkexec][profiling]")
{
  using vkexec::profiling::detail::duration_ns;
  constexpr std::uint32_t k_one_bit = 1;
  constexpr std::uint32_t k_sixteen_bits = 16;
  constexpr std::uint32_t k_thirty_two_bits = 32;
  constexpr std::uint32_t k_forty_eight_bits = 48;
  constexpr std::uint32_t k_sixty_four_bits = 64;
  constexpr float k_period_ns = 2.0F;
  constexpr double k_one_tick_duration_ns = 2.0;
  constexpr double k_three_tick_duration_ns = 6.0;
  constexpr std::uint64_t k_one_bit_start = 1;
  constexpr std::uint64_t k_one_bit_end = 0;
  constexpr std::uint64_t k_sixteen_bit_start = 65534;
  constexpr std::uint64_t k_sixteen_bit_end = 1;
  constexpr std::uint64_t k_thirty_two_bit_start = 0xffff'fffeULL;
  constexpr std::uint64_t k_thirty_two_bit_end = 1;
  constexpr std::uint64_t k_forty_eight_bit_start = (std::uint64_t{ 1 } << k_forty_eight_bits) - 2;
  constexpr std::uint64_t k_forty_eight_bit_end = 1;
  REQUIRE(duration_ns(k_one_bit_start, k_one_bit_end, k_one_bit, k_period_ns) == k_one_tick_duration_ns);
  REQUIRE(duration_ns(k_sixteen_bit_start, k_sixteen_bit_end, k_sixteen_bits, k_period_ns) == k_three_tick_duration_ns);
  REQUIRE(duration_ns(k_thirty_two_bit_start, k_thirty_two_bit_end, k_thirty_two_bits, k_period_ns)
          == k_three_tick_duration_ns);
  REQUIRE(duration_ns(k_forty_eight_bit_start, k_forty_eight_bit_end, k_forty_eight_bits, k_period_ns)
          == k_three_tick_duration_ns);
  REQUIRE(duration_ns(std::numeric_limits<std::uint64_t>::max() - 1, 1, k_sixty_four_bits, k_period_ns)
          == k_three_tick_duration_ns);
  REQUIRE_FALSE(duration_ns(0, 1, 0, 1.0F));
  REQUIRE_FALSE(duration_ns(0, 1, 65, 1.0F));
  REQUIRE_FALSE(duration_ns(0, 1, 64, 0.0F));
}
