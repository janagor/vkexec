#include <catch2/catch_test_macros.hpp>

#include <vkexec/pass.hpp>

#include <cstdint>
#include <limits>

namespace {

constexpr std::uint32_t k_local_size = 64;
constexpr std::uint32_t k_max_work_count = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t k_max_groups_for_local_size = 67108864U;

static_assert(vkexec::dispatch_groups_for(0U, k_local_size).x == 0U);
static_assert(vkexec::dispatch_groups_for(k_local_size + 1U, k_local_size).x == 2U);
static_assert(vkexec::dispatch_groups_for(k_max_work_count, k_local_size).x == k_max_groups_for_local_size);

}// namespace

TEST_CASE("dispatch_groups_for rounds up work count", "[vkexec][dispatch]")
{
  REQUIRE(vkexec::dispatch_groups_for(0, k_local_size).x == 0);
  REQUIRE(vkexec::dispatch_groups_for(1, k_local_size).x == 1);
  REQUIRE(vkexec::dispatch_groups_for(63, k_local_size).x == 1);
  REQUIRE(vkexec::dispatch_groups_for(k_local_size, k_local_size).x == 1);
  REQUIRE(vkexec::dispatch_groups_for(k_local_size + 1, k_local_size).x == 2);
  REQUIRE(vkexec::dispatch_groups_for(k_local_size * 2, k_local_size).x == 2);
  REQUIRE(vkexec::dispatch_groups_for((k_local_size * 2) + 1, k_local_size).x == 3);
}

TEST_CASE("dispatch_groups_for does not overflow when rounding up", "[vkexec][dispatch]")
{
  REQUIRE(vkexec::dispatch_groups_for(k_max_work_count, 1U).x == k_max_work_count);
  REQUIRE(vkexec::dispatch_groups_for(k_max_work_count, 2U).x == 2147483648U);
  REQUIRE(vkexec::dispatch_groups_for(k_max_work_count - 1U, 3U).x == 1431655765U);
  REQUIRE(vkexec::dispatch_groups_for(k_max_work_count, k_local_size).x == k_max_groups_for_local_size);
  REQUIRE(vkexec::dispatch_groups_for(k_max_work_count, k_max_work_count).x == 1U);
}
