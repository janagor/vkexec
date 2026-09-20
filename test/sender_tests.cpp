#include <catch2/catch_test_macros.hpp>

#include <vkexec/error.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec/sync_wait.hpp>

#include <stdexec/execution.hpp>

static_assert(stdexec::sender<vkexec::sender<int>>);
static_assert(stdexec::sender<vkexec::void_sender>);

namespace
{
constexpr auto k_expected_value = 42;
}

TEST_CASE("public sender completes with a value", "[vkexec][sender]")
{
  auto outcome = vkexec::try_sync_wait_value(
    vkexec::make_sender<int>([]() -> vkexec::result<int> { return k_expected_value; }));
  REQUIRE(outcome.has_value());
  REQUIRE(vkexec::expected_take(outcome) == k_expected_value);
}

TEST_CASE("public sender completes with an error", "[vkexec][sender]")
{
  auto outcome = vkexec::try_sync_wait(
    vkexec::make_sender<int>([]() -> vkexec::result<int> { return vkexec::fail(vkexec::errc::unsupported); }));
  REQUIRE(outcome.failed());
  REQUIRE(outcome.take_error().code == vkexec::make_error_code(vkexec::errc::unsupported));
}

TEST_CASE("public void_sender completes successfully", "[vkexec][sender]")
{
  auto outcome = vkexec::try_sync_wait(vkexec::make_void_sender([]() -> vkexec::status { return {}; }));
  REQUIRE(outcome.has_value());
}
