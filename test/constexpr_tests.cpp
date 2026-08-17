#include <catch2/catch_test_macros.hpp>

#include <cstdint>

namespace {

constexpr int k_u8_max = 255;

[[nodiscard]] constexpr auto saturate_u8(int value) noexcept -> std::uint8_t
{
  if (value < 0) { return 0; }
  if (value > k_u8_max) { return static_cast<std::uint8_t>(k_u8_max); }
  return static_cast<std::uint8_t>(value);
}

} // namespace

TEST_CASE("saturate_u8 is usable in constexpr context", "[constexpr]")
{
  STATIC_REQUIRE(saturate_u8(0) == 0);
  STATIC_REQUIRE(saturate_u8(128) == 128);
  STATIC_REQUIRE(saturate_u8(k_u8_max) == k_u8_max);
  STATIC_REQUIRE(saturate_u8(-1) == 0);
  STATIC_REQUIRE(saturate_u8(300) == k_u8_max);
}
