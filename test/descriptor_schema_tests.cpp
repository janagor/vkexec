#include <catch2/catch_test_macros.hpp>

#include <vkexec/bind_resources.hpp>
#include <vkexec/descriptor_schema.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/schema_pass.hpp>

#include <vulkan/vulkan_core.h>

#include <concepts>
#include <cstddef>
#include <span>
#include <utility>

namespace {

using positions = vkexec::storage_buffer<0>;
using velocities = vkexec::storage_buffer<3, vkexec::buffer_access::readonly>;
using sim_schema = vkexec::descriptor_schema<positions, velocities>;
using image_schema =
  vkexec::descriptor_schema<vkexec::storage_image<1>, vkexec::sampled_image<2>, vkexec::sampler_binding<3>>;

struct typed_push_constants
{
  // cppcheck-suppress unusedStructMember
  float scale{ 1.0F };
};

constexpr std::size_t k_fixed_byte_count = 16;
using fixed_byte_span = std::span<std::byte const, k_fixed_byte_count>;

template<class Schema, class... Resources>
concept makes_resource_table = requires(Schema schema, Resources... resources) {
  { vkexec::make_resource_table(schema, std::move(resources)...) } -> std::same_as<vkexec::resource_table>;
};

template<class Schema, class... Resources>
concept makes_schema_bind = requires(Schema schema,
  vkexec::handles::compute_pipeline const &pipe,
  Resources... resources) {
  requires vkexec::detail::sender_adaptor_closure<decltype(vkexec::schema_bind(schema, pipe, std::move(resources)...))>;
};

template<class Push>
concept makes_compute_pass =
  requires(Push push) { vkexec::compute_pass(vkexec::compute_bind{}, push, vkexec::dispatch{}); };

static_assert(sim_schema::binding_count == 2);
static_assert(vkexec::detail::descriptor_schema_slots_unique<positions, velocities>());
static_assert(!vkexec::detail::descriptor_schema_slots_unique<positions, positions>());
static_assert(vkexec::detail::descriptor_schema_slots_sorted<positions, velocities>());
static_assert(!vkexec::detail::descriptor_schema_slots_sorted<velocities, positions>());
static_assert(makes_resource_table<sim_schema, vkexec::resource_ref, vkexec::resource_ref>);
static_assert(!makes_resource_table<sim_schema, vkexec::resource_ref>);
static_assert(makes_schema_bind<sim_schema, vkexec::resource_ref, vkexec::resource_ref>);
static_assert(!makes_schema_bind<sim_schema, vkexec::resource_ref>);
static_assert(vkexec::detail::sender_adaptor_closure<decltype(vkexec::bind_resources(
    std::declval<vkexec::handles::compute_pipeline const &>(),
    std::declval<vkexec::resource_table const &>(),
    typed_push_constants{}))>);
static_assert(vkexec::detail::sender_adaptor_closure<decltype(vkexec::bind_resources(
    std::declval<vkexec::handles::compute_pipeline const &>(),
    std::declval<vkexec::resource_table const &>(),
    std::declval<fixed_byte_span>()))>);
static_assert(makes_compute_pass<typed_push_constants>);
static_assert(!makes_compute_pass<fixed_byte_span>);

}// namespace

TEST_CASE("descriptor_schema builds an ordered resource_table", "[vkexec][descriptor_schema]")
{
  auto const table = vkexec::make_resource_table(
    sim_schema{}, vkexec::buffer_resource(VK_NULL_HANDLE, 64), vkexec::buffer_resource(VK_NULL_HANDLE, 128));

  REQUIRE(table.size() == sim_schema::binding_count);
  REQUIRE(table.entries().front().slot == positions::slot);
  REQUIRE(table.entries().front().resource.byte_size == 64);
  REQUIRE(table.entries().back().slot == velocities::slot);
  REQUIRE(table.entries().back().resource.byte_size == 128);
}

TEST_CASE("descriptor_schema derives explicit compute layout slots", "[vkexec][descriptor_schema]")
{
  constexpr std::size_t k_push_bytes = 16;
  auto const layout = vkexec::layout_desc_from_schema(sim_schema{}, k_push_bytes, { 32, 2, 1 });

  REQUIRE(layout.bindings.size() == sim_schema::binding_count);
  REQUIRE(layout.bindings.front() == vkexec::buffer_access::readwrite);
  REQUIRE(layout.bindings.back() == vkexec::buffer_access::readonly);
  REQUIRE(layout.binding_slots.front() == positions::slot);
  REQUIRE(layout.binding_slots.back() == velocities::slot);
  REQUIRE(layout.push_constant_size == k_push_bytes);
  REQUIRE(layout.local_size.at(0) == 32);
  REQUIRE(layout.local_size.at(1) == 2);
  REQUIRE(layout.local_size.at(2) == 1);
}

TEST_CASE("descriptor_schema assigns image and sampler resource kinds", "[vkexec][descriptor_schema]")
{
  auto const table = vkexec::make_resource_table(image_schema{},
    vkexec::storage_image_resource(VK_NULL_HANDLE),
    vkexec::sampled_image_resource(VK_NULL_HANDLE),
    vkexec::sampler_resource(VK_NULL_HANDLE));
  auto const layout = vkexec::layout_desc_from_schema(image_schema{});

  REQUIRE(table.entries().front().resource.kind == vkexec::resource_kind::storage_image);
  REQUIRE(table.entries().subspan(1).front().resource.kind == vkexec::resource_kind::sampled_image);
  REQUIRE(table.entries().back().resource.kind == vkexec::resource_kind::sampler);
  REQUIRE(layout.binding_kinds.at(0) == vkexec::resource_kind::storage_image);
  REQUIRE(layout.binding_kinds.at(1) == vkexec::resource_kind::sampled_image);
  REQUIRE(layout.binding_kinds.at(2) == vkexec::resource_kind::sampler);
}
