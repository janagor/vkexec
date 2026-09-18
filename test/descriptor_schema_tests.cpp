#include <catch2/catch_test_macros.hpp>

#include <vkexec/descriptor_schema.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>

#include <vulkan/vulkan_core.h>

#include <concepts>
#include <utility>

namespace {

using positions = vkexec::storage_buffer<0>;
using velocities = vkexec::storage_buffer<3, vkexec::buffer_access::readonly>;
using sim_schema = vkexec::descriptor_schema<positions, velocities>;

template<class Schema, class... Resources>
concept makes_resource_table = requires(Schema schema, Resources... resources) {
  { vkexec::make_resource_table(schema, std::move(resources)...) } -> std::same_as<vkexec::resource_table>;
};

static_assert(sim_schema::binding_count == 2);
static_assert(vkexec::detail::descriptor_schema_slots_unique<positions, velocities>());
static_assert(!vkexec::detail::descriptor_schema_slots_unique<positions, positions>());
static_assert(vkexec::detail::descriptor_schema_slots_sorted<positions, velocities>());
static_assert(!vkexec::detail::descriptor_schema_slots_sorted<velocities, positions>());
static_assert(makes_resource_table<sim_schema, vkexec::resource_ref, vkexec::resource_ref>);
static_assert(!makes_resource_table<sim_schema, vkexec::resource_ref>);

}// namespace

TEST_CASE("descriptor_schema builds an ordered resource_table", "[vkexec][descriptor_schema]")
{
  auto const table = vkexec::make_resource_table(sim_schema{},
    vkexec::resource_ref{ .buffer = VK_NULL_HANDLE, .byte_size = 64 },
    vkexec::resource_ref{ .buffer = VK_NULL_HANDLE, .byte_size = 128 });

  REQUIRE(table.size() == sim_schema::binding_count);
  REQUIRE(table.entries().front().slot == positions::slot);
  REQUIRE(table.entries().front().resource.byte_size == 64);
  REQUIRE(table.entries().back().slot == velocities::slot);
  REQUIRE(table.entries().back().resource.byte_size == 128);
}
