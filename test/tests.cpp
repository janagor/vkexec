#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>

TEST_CASE("vkexec headers compile", "[vkexec]")
{
  vkexec::scheduler_options const opts{};
  (void)opts;
  REQUIRE(true);
}
