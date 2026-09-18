#include <catch2/catch_test_macros.hpp>

#include <vkexec/resource_table.hpp>

#include <vulkan/vulkan_core.h>

TEST_CASE("resource_table preserves logical storage bindings", "[vkexec][resource_table]")
{
  auto const table = vkexec::bindings(
    vkexec::resource_binding{ .slot = 3, .resource = { .buffer = VK_NULL_HANDLE, .byte_size = 64 } },
    vkexec::resource_binding{ .slot = 7, .resource = { .buffer = VK_NULL_HANDLE, .byte_size = 128 } });

  REQUIRE(table.size() == 2);
  REQUIRE(table.entries().front().slot == 3);
  REQUIRE(table.entries().front().resource.byte_size == 64);
  REQUIRE(table.entries().back().slot == 7);
  REQUIRE(table.entries().back().resource.byte_size == 128);
}
