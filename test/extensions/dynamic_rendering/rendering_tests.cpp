#include <catch2/catch_test_macros.hpp>
#include <vkexec_extensions/dynamic_rendering/rendering.hpp>

TEST_CASE("dynamic rendering defaults to one layer", "[vkexec][dynamic_rendering]")
{
  vkexec::rendering_info const info{};
  REQUIRE(info.layer_count == 1);
  REQUIRE(info.depth == nullptr);
  REQUIRE(info.color.empty());
}
