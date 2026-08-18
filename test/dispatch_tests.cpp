#include <catch2/catch_test_macros.hpp>

#include <vkexec/pass.hpp>

#include <cstdint>

namespace {

constexpr std::uint32_t k_local_size = 64;

}// namespace

TEST_CASE("dispatch_groups_for rounds up work count", "[vkexec][dispatch]")
{
  REQUIRE(vkexec::dispatch_groups_for(0, k_local_size).x == 0);
  REQUIRE(vkexec::dispatch_groups_for(1, k_local_size).x == 1);
  REQUIRE(vkexec::dispatch_groups_for(k_local_size, k_local_size).x == 1);
  REQUIRE(vkexec::dispatch_groups_for(k_local_size + 1, k_local_size).x == 2);
  REQUIRE(vkexec::dispatch_groups_for(k_local_size * 2, k_local_size).x == 2);
}

TEST_CASE("dispatch_groups_for treats zero local size as one", "[vkexec][dispatch]")
{
  REQUIRE(vkexec::dispatch_groups_for(0, 0).x == 0);
  REQUIRE(vkexec::dispatch_groups_for(1, 0).x == 1);
  REQUIRE(vkexec::dispatch_groups_for(100, 0).x == 100);
}
