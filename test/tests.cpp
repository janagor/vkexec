#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/descriptor_strategy.hpp>

#include <concepts>

#include <stdexec/execution.hpp>

static_assert(std::same_as<decltype(vkexec::descriptor_sets), vkexec::descriptor_sets_t const>);
static_assert(stdexec::sender<decltype(vkexec::factory::make_context())>);
static_assert(std::movable<vkexec::queue_guard>);
static_assert(!std::copy_constructible<vkexec::queue_guard>);

TEST_CASE("vkexec headers compile", "[vkexec]")
{
  vkexec::scheduler_options const opts{};
  (void)opts;
  REQUIRE(true);
}
