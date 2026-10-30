#include <catch2/catch_test_macros.hpp>
#include <vkexec/detail/resource_state_tracker_validation.hpp>

using namespace vkexec;
using namespace vkexec::detail;

TEST_CASE("clamping checks both extent boundaries before changing open ends", "[vkexec][resource-helpers]")
{
  for (auto range : { buffer_range{ 0, 128 },
         buffer_range{ 0, 129 },
         buffer_range{ 128, 129 },
         buffer_range{ 128, {} },
         buffer_range{ 32, {} } }) {
    auto original = range;
    bool ok = original.begin < 128 && (!original.end || *original.end <= 128);
    REQUIRE(static_cast<bool>(clamp_buffer_range(range, 128)) == ok);
    if (ok) {
      REQUIRE(range.end == 128);
      REQUIRE(range.begin == original.begin);
    } else {
      REQUIRE(range == original);
    }
  }
  image_extent extent{ .mip_levels = 4, .array_layers = 4 };
  for (auto mips : { subresource_interval{ 0, 4 },
         subresource_interval{ 0, 5 },
         subresource_interval{ 4, 5 },
         subresource_interval{ 4, {} },
         subresource_interval{ 1, {} } }) {
    for (auto layers : { subresource_interval{ 0, 4 },
           subresource_interval{ 0, 5 },
           subresource_interval{ 4, 5 },
           subresource_interval{ 4, {} },
           subresource_interval{ 1, {} } }) {
      image_region region{ VK_IMAGE_ASPECT_COLOR_BIT, mips, layers };
      auto original = region;
      bool ok =
        mips.begin < 4 && (!mips.end || *mips.end <= 4) && layers.begin < 4 && (!layers.end || *layers.end <= 4);
      REQUIRE(static_cast<bool>(clamp_image_region(region, extent)) == ok);
      if (ok) {
        REQUIRE(region.mips.end == 4);
        REQUIRE(region.layers.end == 4);
        REQUIRE(region.mips.begin == mips.begin);
        REQUIRE(region.layers.begin == layers.begin);
      } else {
        REQUIRE(region == original);
      }
    }
  }
}

TEST_CASE("source scopes select active readers by queue and preserve all mask bits", "[vkexec][resource-helpers]")
{
  int a{}, b{};
  auto q = static_cast<VkQueue>(static_cast<void *>(&a));
  auto other = static_cast<VkQueue>(static_cast<void *>(&b));
  access_frontier<tracked_buffer> f;
  REQUIRE(frontier_handoff_queue(f) == VK_NULL_HANDLE);
  REQUIRE(source_scope(f, q) == usage_scope{});
  tracked_buffer anchor{ .scope = { VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT }, .queue = q };
  f.anchor = anchor;
  REQUIRE(frontier_handoff_queue(f) == q);
  REQUIRE(source_scope(f, q) == anchor.scope);
  REQUIRE(source_scope(f, other) == usage_scope{});
  f.readers.push_back(
    { .scope = { VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT }, .queue = q });
  REQUIRE(frontier_handoff_queue(f) == q);
  REQUIRE(source_scope(f, q) == f.readers[0].scope);
  f.readers.push_back(
    { .scope = { VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT }, .queue = other });
  f.readers.push_back({ .scope = { VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT } });
  REQUIRE(frontier_handoff_queue(f) == VK_NULL_HANDLE);
  auto check = [&](VkQueue selected, VkPipelineStageFlags2 stage, VkAccessFlags2 access) {
    auto scope = source_scope(f, selected);
    REQUIRE(scope.stage == stage);
    REQUIRE(scope.access == access);
  };
  check(q,
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
  check(other,
    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
    VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
  check(VK_NULL_HANDLE,
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
      | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
  f.readers.pop_back();
  REQUIRE(frontier_handoff_queue(f) == other);
}

TEST_CASE("buffer conflicts isolate overlap step access family queue and anchor", "[vkexec][resource-helpers]")
{
  int a{}, b{};
  auto q = static_cast<VkQueue>(static_cast<void *>(&a));
  auto other = static_cast<VkQueue>(static_cast<void *>(&b));
  for (bool overlap : { false, true })
    for (auto prior : { resource_access::read, resource_access::write })
      for (auto next : { resource_access::read, resource_access::write })
        for (auto queue : { q, other })
          for (unsigned family : { 2U, 3U })
            for (std::size_t step : { 0U, 1U }) {
              tracked_buffer access{ .access = prior, .scope = {}, .step = 0, .family = 2, .queue = q };
              std::vector<buffer_region_state> states{ { .range = { 0, 16 },
                .accesses = { .anchor = {}, .readers = { access } } } };
              buffer_use declaration{ .access = next };
              bool expected =
                !overlap || step != 0
                || (prior == resource_access::read && next == resource_access::read && queue == q && family == 2);
              REQUIRE(
                static_cast<bool>(validate_buffer_use_conflicts(
                  states, overlap ? buffer_range{ 8, 24 } : buffer_range{ 16, 24 }, declaration, step, family, queue))
                == expected);
              states[0].accesses = { .anchor = access, .readers = {} };
              REQUIRE(
                static_cast<bool>(validate_buffer_use_conflicts(states, { 8, 24 }, declaration, step, family, queue))
                == (step != 0));
            }
}

TEST_CASE(
  "image conflicts require compatible layout ownership queue and "
  "effective final layout",
  "[vkexec][resource-helpers]")
{
  int a{}, b{};
  auto q = static_cast<VkQueue>(static_cast<void *>(&a));
  auto other = static_cast<VkQueue>(static_cast<void *>(&b));
  image_region region{ VK_IMAGE_ASPECT_COLOR_BIT, { 0, 1 }, { 0, 1 } };
  usage_scope scope{ .layout = VK_IMAGE_LAYOUT_GENERAL };
  std::vector<image_region_state> states{ { .region = region, .layout = scope.layout, .family = 2, .accesses = {} } };
  image_use declaration{ .access = resource_access::read };
  REQUIRE_FALSE(validate_image_use_conflicts(states, region, declaration, 0, 2, q, true, scope, scope, false));
  declaration.initial_layout = VK_IMAGE_LAYOUT_GENERAL;
  REQUIRE(validate_image_use_conflicts(states, region, declaration, 0, 2, q, true, scope, scope, false));
  declaration.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  declaration.access = resource_access::write;
  REQUIRE(validate_image_use_conflicts(states, region, declaration, 0, 2, q, true, scope, scope, false));
  for (auto prior : { resource_access::read, resource_access::write })
    for (auto next : { resource_access::read, resource_access::write })
      for (bool same_layout : { false, true })
        for (bool same_final : { false, true })
          for (bool transition : { false, true })
            for (bool exclusive : { false, true })
              for (unsigned family : { 2U, 3U })
                for (auto queue : { q, other })
                  for (std::size_t step : { 0U, 1U }) {
                    tracked_image access{ .access = prior, .scope = {}, .step = 0, .family = 2, .queue = q };
                    states[0].accesses = { .anchor = {}, .readers = { access } };
                    declaration.access = next;
                    auto after = scope;
                    if (!same_final) after.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    auto next_scope = scope;
                    if (!same_layout) next_scope.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    bool compatible = next == resource_access::read && prior == resource_access::read
                                      && (!exclusive || family == 2) && same_layout && same_final && !transition
                                      && queue == q;
                    REQUIRE(
                      static_cast<bool>(validate_image_use_conflicts(
                        states, region, declaration, step, family, queue, exclusive, next_scope, after, transition))
                      == (step != 0 || compatible));
                    states[0].accesses = { .anchor = access, .readers = {} };
                    REQUIRE(
                      static_cast<bool>(validate_image_use_conflicts(
                        states, region, declaration, step, family, queue, exclusive, next_scope, after, transition))
                      == (step != 0));
                    auto disjoint = region;
                    disjoint.mips = { 1, 2 };
                    REQUIRE(validate_image_use_conflicts(
                      states, disjoint, declaration, 0, family, queue, exclusive, next_scope, after, transition));
                  }
}

TEST_CASE("mixed aspect validation distinguishes overlap from disjoint masks", "[vkexec][resource-helpers]")
{
  auto depth = VK_IMAGE_ASPECT_DEPTH_BIT;
  auto stencil = VK_IMAGE_ASPECT_STENCIL_BIT;
  for (VkImageAspectFlags existing :
    { VkImageAspectFlags(depth), VkImageAspectFlags(stencil), VkImageAspectFlags(depth | stencil) })
    for (VkImageAspectFlags next :
      { VkImageAspectFlags(depth), VkImageAspectFlags(stencil), VkImageAspectFlags(depth | stencil) }) {
      image_region region{ next, { 0, 1 }, { 0, 1 } };
      REQUIRE(reject_mixed_aspect_declarations({}, region));
      REQUIRE(static_cast<bool>(reject_mixed_aspect_declarations(
                { { .region = { existing, { 0, 1 }, { 0, 1 } }, .accesses = {} } }, region))
              == ((existing & next) == 0 || existing == next));
    }
}
