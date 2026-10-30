#include <array>
#include <catch2/catch_test_macros.hpp>
#include <vkexec/detail/resource_state_tracker_helpers.hpp>

using namespace vkexec;
using namespace vkexec::detail;

TEST_CASE("usage validation covers every usage and access combination", "[vkexec][resource-helpers]")
{
  for (auto access : { resource_access::read, resource_access::write, resource_access::read_write }) {
    for (auto usage : { image_usage::sampled_compute,
           image_usage::sampled_fragment,
           image_usage::sampled_graphics,
           image_usage::transfer_source })
      REQUIRE(valid(image_use{ .usage = usage, .access = access }) == (access == resource_access::read));
    REQUIRE(valid(image_use{ .usage = image_usage::transfer_destination, .access = access })
            == (access == resource_access::write));
    for (auto usage : { image_usage::storage_compute,
           image_usage::storage_graphics,
           image_usage::color_attachment,
           image_usage::depth_attachment })
      REQUIRE(valid(image_use{ .usage = usage, .access = access }));
    for (auto usage : { buffer_usage::uniform_compute,
           buffer_usage::uniform_vertex,
           buffer_usage::uniform_fragment,
           buffer_usage::vertex,
           buffer_usage::index,
           buffer_usage::indirect,
           buffer_usage::transfer_source })
      REQUIRE(valid(buffer_use{ .usage = usage, .access = access }) == (access == resource_access::read));
    REQUIRE(valid(buffer_use{ .usage = buffer_usage::transfer_destination, .access = access })
            == (access == resource_access::write));
    for (auto usage : { buffer_usage::storage_compute, buffer_usage::storage_graphics })
      REQUIRE(valid(buffer_use{ .usage = usage, .access = access }));
  }
  REQUIRE(access_mask(resource_access::read, 0x10, 0x80) == 0x10);
  REQUIRE(access_mask(resource_access::write, 0x10, 0x80) == 0x80);
  REQUIRE(access_mask(resource_access::read_write, 0x10, 0x80) == 0x90);
}

TEST_CASE("image scope mapping has exact Vulkan stages accesses and layouts", "[vkexec][resource-helpers]")
{
  struct Row
  {
    image_usage usage;
    VkPipelineStageFlags2 stage;
    VkAccessFlags2 read, write;
    VkImageLayout layout;
    bool fixed;
  };
  std::array rows{ Row{ image_usage::sampled_compute,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     0,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     true },
    Row{ image_usage::sampled_fragment,
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
      0,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      true },
    Row{ image_usage::sampled_graphics,
      VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
      0,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      true },
    Row{ image_usage::storage_compute,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      VK_IMAGE_LAYOUT_GENERAL,
      false },
    Row{ image_usage::storage_graphics,
      VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
      VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      VK_IMAGE_LAYOUT_GENERAL,
      false },
    Row{ image_usage::color_attachment,
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      false },
    Row{ image_usage::depth_attachment,
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      false },
    Row{ image_usage::transfer_source,
      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
      VK_ACCESS_2_TRANSFER_READ_BIT,
      0,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      true },
    Row{ image_usage::transfer_destination,
      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
      VK_ACCESS_2_TRANSFER_WRITE_BIT,
      0,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      true } };
  for (auto row : rows)
    for (auto access : { resource_access::read, resource_access::write, resource_access::read_write }) {
      image_use use{ .usage = row.usage, .access = access };
      auto expected = row.fixed                          ? row.read
                      : access == resource_access::read  ? row.read
                      : access == resource_access::write ? row.write
                                                         : row.read | row.write;
      auto scope = default_image_scope(use);
      REQUIRE(scope.stage == row.stage);
      REQUIRE(scope.access == expected);
      REQUIRE(scope.layout == row.layout);
      REQUIRE(image_scope(use) == scope);
      use.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      scope.layout = use.layout;
      REQUIRE(image_scope(use) == scope);
    }
}

TEST_CASE("buffer scope mapping has exact Vulkan stages and accesses", "[vkexec][resource-helpers]")
{
  struct Row
  {
    buffer_usage usage;
    VkPipelineStageFlags2 stage;
    VkAccessFlags2 read, write;
  };
  std::array rows{
    Row{ buffer_usage::indirect, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, 0 },
    Row{
      buffer_usage::vertex, VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT, 0 },
    Row{ buffer_usage::index, VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT, VK_ACCESS_2_INDEX_READ_BIT, 0 },
    Row{ buffer_usage::uniform_compute, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT, 0 },
    Row{ buffer_usage::uniform_vertex, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT, 0 },
    Row{ buffer_usage::uniform_fragment, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT, 0 },
    Row{ buffer_usage::transfer_source, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, 0 },
    Row{ buffer_usage::transfer_destination, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, 0 },
    Row{ buffer_usage::storage_compute,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT },
    Row{ buffer_usage::storage_graphics,
      VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
      VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT }
  };
  for (auto row : rows)
    for (auto access : { resource_access::read, resource_access::write, resource_access::read_write }) {
      auto scope = buffer_scope({ .usage = row.usage, .access = access });
      auto expected = row.write == 0                     ? row.read
                      : access == resource_access::read  ? row.read
                      : access == resource_access::write ? row.write
                                                         : row.read | row.write;
      REQUIRE(scope.stage == row.stage);
      REQUIRE(scope.access == expected);
      REQUIRE(scope.layout == VK_IMAGE_LAYOUT_UNDEFINED);
    }
}

TEST_CASE(
  "buffer partitioning preserves complete metadata and merges only "
  "identical frontiers",
  "[vkexec][resource-helpers]")
{
  auto empty = partition_buffer_regions({}, { 0, 32 });
  REQUIRE(empty);
  REQUIRE(empty->size() == 1);
  REQUIRE(empty->at(0).range == buffer_range{ 0, 32 });
  REQUIRE(empty->at(0).accesses == access_frontier<tracked_buffer>{});
  REQUIRE(empty->at(0).family == VK_QUEUE_FAMILY_IGNORED);
  buffer_region_state state{ .range = { 8, 24 },
    .family = 2,
    .accesses = { .anchor = tracked_buffer{ .scope = {}, .step = 1 },
      .readers = { { .scope = {}, .step = 2 }, { .scope = {}, .step = 3 } } } };
  auto outer = partition_buffer_regions({ state }, { 0, 32 });
  REQUIRE(outer);
  REQUIRE(outer->size() == 3);
  REQUIRE(outer->at(0).range == buffer_range{ 0, 8 });
  REQUIRE(outer->at(1).range == state.range);
  REQUIRE(outer->at(1).accesses == state.accesses);
  REQUIRE(outer->at(1).family == 2);
  REQUIRE(outer->at(2).range == buffer_range{ 24, 32 });
  REQUIRE(outer->at(0).accesses == access_frontier<tracked_buffer>{});
  REQUIRE(outer->at(2).accesses == access_frontier<tracked_buffer>{});
  auto inner = partition_buffer_regions({ state }, { 12, 16 });
  REQUIRE(inner);
  REQUIRE(inner->size() == 3);
  for (auto const &piece : *inner) {
    REQUIRE(piece.accesses == state.accesses);
    REQUIRE(piece.family == state.family);
  }
  REQUIRE(inner->at(0).range == buffer_range{ 8, 12 });
  REQUIRE(inner->at(1).range == buffer_range{ 12, 16 });
  REQUIRE(inner->at(2).range == buffer_range{ 16, 24 });
  merge_buffer_regions(*inner);
  REQUIRE(inner->size() == 1);
  REQUIRE(inner->at(0).range == state.range);
  merge_buffer_regions(*inner);
  REQUIRE(inner->size() == 1);
  REQUIRE(inner->at(0).accesses == state.accesses);
  for (unsigned change = 0; change < 4; ++change) {
    auto right = state;
    right.range = { 24, 32 };
    if (change == 1) ++right.family;
    if (change == 2) ++right.accesses.anchor->step;
    if (change == 3) ++right.accesses.readers[0].step;
    std::vector regions{ state, right };
    merge_buffer_regions(regions);
    REQUIRE(regions.size() == (change == 0 ? 1 : 2));
    if (change != 0) {
      REQUIRE(regions[0].accesses == state.accesses);
      REQUIRE(regions[1].accesses == right.accesses);
      REQUIRE(regions[1].family == right.family);
    }
  }
  state.range = { 0, {} };
  REQUIRE_FALSE(partition_buffer_regions({ state }, { 0, 16 }));
  state.range = { 0, 16 };
  REQUIRE_FALSE(partition_buffer_regions({ state }, { 0, {} }));
}

TEST_CASE(
  "image partition grid preserves every cell frontier layout owner and "
  "aspect",
  "[vkexec][resource-helpers]")
{
  image_region_state state{ .region = { VK_IMAGE_ASPECT_COLOR_BIT, { 0, 4 }, { 0, 4 } },
    .layout = VK_IMAGE_LAYOUT_GENERAL,
    .family = 2,
    .accesses = { .anchor = tracked_image{ .scope = {}, .step = 1 },
      .readers = { { .scope = {}, .step = 2 }, { .scope = {}, .step = 3 } } } };
  for (unsigned mb = 0; mb < 4; ++mb)
    for (unsigned me = mb + 1; me <= 4; ++me)
      for (unsigned lb = 0; lb < 4; ++lb)
        for (unsigned le = lb + 1; le <= 4; ++le) {
          auto pieces = partition_image_regions({ state }, { VK_IMAGE_ASPECT_COLOR_BIT, { mb, me }, { lb, le } });
          REQUIRE(pieces);
          auto check = [&] {
            std::array<unsigned, 16> cells{};
            for (auto const &piece : *pieces) {
              REQUIRE(piece.accesses == state.accesses);
              REQUIRE(piece.layout == state.layout);
              REQUIRE(piece.family == state.family);
              REQUIRE(piece.region.aspects == state.region.aspects);
              for (auto m = piece.region.mips.begin; m < *piece.region.mips.end; ++m)
                for (auto l = piece.region.layers.begin; l < *piece.region.layers.end; ++l) {
                  REQUIRE(m < 4);
                  REQUIRE(l < 4);
                  ++cells.at(m * 4 + l);
                }
            }
            for (auto count : cells) REQUIRE(count == 1);
          };
          check();
          merge_image_regions(*pieces);
          check();
          REQUIRE(pieces->size() == 1);
          REQUIRE(pieces->at(0).region == state.region);
          merge_image_regions(*pieces);
          check();
          REQUIRE(pieces->size() == 1);
        }
  for (unsigned change = 0; change < 6; ++change) {
    auto left = state;
    left.region.mips = { 0, 1 };
    auto right = left;
    right.region.mips = { 1, 2 };
    if (change == 1) right.region.aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (change == 2) right.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (change == 3) ++right.family;
    if (change == 4) ++right.accesses.anchor->step;
    if (change == 5) ++right.accesses.readers[0].step;
    std::vector regions{ left, right };
    merge_image_regions(regions);
    REQUIRE(regions.size() == (change == 0 ? 1 : 2));
    if (change != 0) {
      REQUIRE(regions[0].region == left.region);
      REQUIRE(regions[1].region == right.region);
      REQUIRE(regions[1].accesses == right.accesses);
      REQUIRE(regions[1].family == right.family);
      REQUIRE(regions[1].layout == right.layout);
    }
  }
  auto open = state;
  open.region.mips.end.reset();
  REQUIRE_FALSE(partition_image_regions({ open }, { VK_IMAGE_ASPECT_COLOR_BIT, { 0, 1 }, { 0, 4 } }));
  open = state;
  open.region.layers.end.reset();
  REQUIRE_FALSE(partition_image_regions({ open }, { VK_IMAGE_ASPECT_COLOR_BIT, { 0, 4 }, { 0, 1 } }));
}

#if defined(VKEXEC_ENABLE_ALLOCATION_FAULT_INJECTION) && VKEXEC_HAS_EXCEPTIONS
TEST_CASE(
  "allocation fault seam stays disarmed and throws on the exact armed "
  "checkpoint",
  "[vkexec][resource-helpers]")
{
  struct Reset
  {
    ~Reset() { resource_allocation_fault::disarm(); }
  } reset;
  resource_allocation_fault::disarm();
  resource_allocation_fault::maybe_throw();
  REQUIRE(resource_allocation_fault::remaining == 0);
  resource_allocation_fault::arm(2);
  resource_allocation_fault::maybe_throw();
  REQUIRE(resource_allocation_fault::remaining == 1);
  REQUIRE_THROWS_AS(resource_allocation_fault::maybe_throw(), std::bad_alloc);
  REQUIRE(resource_allocation_fault::remaining == 0);
}
#endif

TEST_CASE(
  "image partition reference distinguishes old states from newly "
  "covered holes",
  "[vkexec][resource-helpers]")
{
  std::vector<image_region_state> states;
  std::array<std::optional<image_region_state>, 16> reference{};
  for (auto rect : { image_region{ VK_IMAGE_ASPECT_COLOR_BIT, { 1, 3 }, { 1, 3 } },
         image_region{ VK_IMAGE_ASPECT_COLOR_BIT, { 0, 4 }, { 0, 4 } },
         image_region{ VK_IMAGE_ASPECT_COLOR_BIT, { 0, 1 }, { 0, 4 } },
         image_region{ VK_IMAGE_ASPECT_COLOR_BIT, { 0, 4 }, { 3, 4 } } }) {
    auto partitioned = partition_image_regions(states, rect);
    REQUIRE(partitioned);
    for (auto const &piece : *partitioned)
      for (unsigned m = piece.region.mips.begin; m < *piece.region.mips.end; ++m)
        for (unsigned l = piece.region.layers.begin; l < *piece.region.layers.end; ++l) {
          auto const &old = reference.at(m * 4 + l);
          if (old) {
            REQUIRE(piece.family == old->family);
            REQUIRE(piece.layout == old->layout);
            REQUIRE(piece.accesses == old->accesses);
          } else {
            REQUIRE(piece.family == VK_QUEUE_FAMILY_IGNORED);
            REQUIRE(piece.layout == VK_IMAGE_LAYOUT_UNDEFINED);
            REQUIRE(piece.accesses == access_frontier<tracked_image>{});
          }
        }
    auto generation = static_cast<unsigned>(states.size() + 1);
    for (auto &piece : *partitioned)
      if (overlaps(piece.region, rect)) {
        piece.family = generation;
        piece.layout = VK_IMAGE_LAYOUT_GENERAL;
        piece.accesses = { .anchor = {}, .readers = { { .scope = {}, .step = generation } } };
      }
    for (unsigned m = rect.mips.begin; m < *rect.mips.end; ++m)
      for (unsigned l = rect.layers.begin; l < *rect.layers.end; ++l)
        reference.at(m * 4 + l) = image_region_state{ .region = {},
          .layout = VK_IMAGE_LAYOUT_GENERAL,
          .family = generation,
          .accesses = { .anchor = {}, .readers = { { .scope = {}, .step = generation } } } };
    states = std::move(*partitioned);
    merge_image_regions(states);
    std::array<unsigned, 16> counts{};
    for (auto const &piece : states)
      for (unsigned m = piece.region.mips.begin; m < *piece.region.mips.end; ++m)
        for (unsigned l = piece.region.layers.begin; l < *piece.region.layers.end; ++l) {
          auto const &expected = reference.at(m * 4 + l);
          REQUIRE(expected);
          REQUIRE(piece.family == expected->family);
          REQUIRE(piece.layout == expected->layout);
          REQUIRE(piece.accesses == expected->accesses);
          ++counts.at(m * 4 + l);
        }
    for (unsigned i = 0; i < 16; ++i) REQUIRE(counts[i] == (reference[i] ? 1 : 0));
  }
}
