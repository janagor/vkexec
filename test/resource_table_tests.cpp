#include <catch2/catch_test_macros.hpp>

#include <vkexec/resource_table.hpp>

#include <vulkan/vulkan_core.h>

TEST_CASE("resource_table preserves logical storage bindings", "[vkexec][resource_table]")
{
  auto const table = vkexec::bindings(
    vkexec::resource_binding{ .slot = 3, .resource = vkexec::buffer_resource(VK_NULL_HANDLE, 64) },
    vkexec::resource_binding{ .slot = 7, .resource = vkexec::buffer_resource(VK_NULL_HANDLE, 128) });

  REQUIRE(table.size() == 2);
  REQUIRE(table.entries().front().slot == 3);
  REQUIRE(table.entries().front().resource.byte_size == 64);
  REQUIRE(table.entries().back().slot == 7);
  REQUIRE(table.entries().back().resource.byte_size == 128);
}

TEST_CASE("resource_table represents image and sampler kinds", "[vkexec][resource_table]")
{
  auto const table = vkexec::bindings(
    vkexec::resource_binding{ .slot = 1, .resource = vkexec::storage_image_resource(VK_NULL_HANDLE) },
    vkexec::resource_binding{ .slot = 2, .resource = vkexec::sampled_image_resource(VK_NULL_HANDLE) },
    vkexec::resource_binding{ .slot = 3, .resource = vkexec::sampler_resource(VK_NULL_HANDLE) });

  REQUIRE(table.entries().front().resource.kind == vkexec::resource_kind::storage_image);
  REQUIRE(table.entries().subspan(1).front().resource.kind == vkexec::resource_kind::sampled_image);
  REQUIRE(table.entries().back().resource.kind == vkexec::resource_kind::sampler);
}
