#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/descriptor_strategy.hpp>
#include <vkexec/sender.hpp>

#include <concepts>
#include <memory>

static_assert(std::same_as<decltype(vkexec::descriptor_sets), vkexec::descriptor_sets_t const>);
static_assert(
  std::same_as<decltype(vkexec::factory::make_context()), vkexec::sender<std::unique_ptr<vkexec::context>>>);

TEST_CASE("vkexec headers compile", "[vkexec]")
{
  vkexec::scheduler_options const opts{};
  (void)opts;
  REQUIRE(true);
}
