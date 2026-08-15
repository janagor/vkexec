#include <catch2/catch_test_macros.hpp>

#include <myproject/sample_library.hpp>

TEST_CASE("Factorials are computed with constexpr", "[factorial]")
{
  STATIC_REQUIRE(FactorialConstexpr(0) == 1);
  STATIC_REQUIRE(FactorialConstexpr(1) == 1);
  STATIC_REQUIRE(FactorialConstexpr(2) == 2);
  STATIC_REQUIRE(FactorialConstexpr(3) == 6);
  STATIC_REQUIRE(FactorialConstexpr(10) == 3628800);
}
