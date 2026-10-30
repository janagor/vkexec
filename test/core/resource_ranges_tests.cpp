// NOLINTBEGIN(readability-function-cognitive-complexity)

#include "vkexec/detail/resource_state_tracker_helpers.hpp"
#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier_params.hpp>
#include <vkexec/detail/resource_barrier_coalescing.hpp>
#include <vkexec/detail/resource_ranges.hpp>
#include <vkexec/resource_use.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace {

constexpr VkDeviceSize k_begin = 0;
constexpr VkDeviceSize k_quarter = 128;
constexpr VkDeviceSize k_middle = 256;
constexpr VkDeviceSize k_three_quarters = 384;
constexpr VkDeviceSize k_end = 512;
constexpr std::size_t k_split_count = 3;
constexpr std::size_t k_one_region = 1;
constexpr std::size_t k_first_step = 0;
constexpr std::size_t k_second_step = 1;
constexpr std::uint32_t k_family = 0;

}// namespace

TEST_CASE("open remainder requires a positive cut", "[vkexec][ranges]")
{
  using vkexec::detail::ambiguous_open_remainder;
  using vkexec::detail::buffer_range;

  buffer_range const open{ .begin = 128, .end = std::nullopt };
  REQUIRE_FALSE(ambiguous_open_remainder(open, VkDeviceSize{ 128 }));
  REQUIRE(ambiguous_open_remainder(open, VkDeviceSize{ 129 }));

  buffer_range const finite{ .begin = 128, .end = 256 };
  REQUIRE_FALSE(ambiguous_open_remainder(finite, VkDeviceSize{ 129 }));
}

TEST_CASE("subresource normalization accepts exact maximum end", "[vkexec][ranges]")
{
  using vkexec::detail::normalize_subresources;
  constexpr auto max = std::numeric_limits<std::uint32_t>::max();

  auto const valid = normalize_subresources(max - 4u, 4u, VK_REMAINING_MIP_LEVELS);
  REQUIRE(valid.has_value());
  REQUIRE(valid->begin == max - 4u);
  REQUIRE(valid->end.has_value());
  REQUIRE(*valid->end == max);

  auto const overflow = normalize_subresources(max - 4u, 5u, VK_REMAINING_MIP_LEVELS);
  REQUIRE_FALSE(overflow.has_value());
}

TEST_CASE("buffer interval algebra uses half-open bounds", "[vkexec][ranges]")
{
  using vkexec::detail::adjacent;
  using vkexec::detail::buffer_range;
  using vkexec::detail::contains;
  using vkexec::detail::intersection;
  using vkexec::detail::overlaps;
  using vkexec::detail::region_size;
  using vkexec::detail::split;

  buffer_range const first{ .begin = k_begin, .end = k_middle };
  buffer_range const partial{ .begin = k_quarter, .end = k_three_quarters };
  buffer_range const tail{ .begin = k_middle, .end = std::nullopt };
  REQUIRE(overlaps(first, partial));
  REQUIRE_FALSE(overlaps(first, tail));
  REQUIRE(adjacent(first, tail));
  REQUIRE(intersection(first, partial) == buffer_range{ .begin = k_quarter, .end = k_middle });
  REQUIRE_FALSE(intersection(first, tail));
  REQUIRE(contains(first, first));
  REQUIRE(contains(first, buffer_range{ .begin = k_begin, .end = k_quarter }));
  REQUIRE(contains(tail, tail));
  REQUIRE(contains(tail, buffer_range{ .begin = k_three_quarters, .end = k_end }));
  REQUIRE_FALSE(contains(first, tail));
  REQUIRE(region_size(tail) == VK_WHOLE_SIZE);
  REQUIRE(region_size(first) == k_middle);
  auto const pieces = split(tail, k_three_quarters);
  REQUIRE(pieces.has_value());
  REQUIRE(pieces->first == buffer_range{ .begin = k_middle, .end = k_three_quarters });
  REQUIRE(pieces->second == buffer_range{ .begin = k_three_quarters, .end = std::nullopt });
  REQUIRE_FALSE(split(first, k_begin));
  REQUIRE_FALSE(split(first, k_middle));
}

TEST_CASE("buffer normalization validates finite ranges without adding whole size", "[vkexec][ranges]")
{
  using vkexec::detail::normalize;

  vkexec::buffer_use use{ .offset = k_quarter, .size = k_middle };
  auto range = normalize(use);
  REQUIRE(range.has_value());
  REQUIRE(range->begin == k_quarter);
  REQUIRE(range->end == k_three_quarters);
  use.size = VK_WHOLE_SIZE;
  range = normalize(use);
  REQUIRE(range.has_value());
  REQUIRE_FALSE(range->end);
  use.size = 0;
  REQUIRE_FALSE(normalize(use));
  use.offset = std::numeric_limits<VkDeviceSize>::max() - k_quarter;
  use.size = k_middle;
  REQUIRE_FALSE(normalize(use));
  use.size = k_quarter;
  REQUIRE(normalize(use));
  use.offset = std::numeric_limits<VkDeviceSize>::max();
  use.size = VK_WHOLE_SIZE;
  REQUIRE_FALSE(normalize(use));
}

TEST_CASE("partitioning clones complete frontiers and merging preserves them", "[vkexec][ranges][resource-helpers]")
{
  using vkexec::detail::buffer_region_state;
  using vkexec::detail::merge_buffer_regions;
  using vkexec::detail::partition_buffer_regions;
  using vkexec::detail::tracked_buffer;

  tracked_buffer const writer{ .access = vkexec::resource_access::write,
    .scope = { .stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT },
    .step = k_first_step,
    .family = k_family };
  tracked_buffer const reader{ .access = vkexec::resource_access::read,
    .scope = { .stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_READ_BIT },
    .step = k_second_step,
    .family = k_family };
  std::vector const original{ buffer_region_state{ .range = { .begin = k_begin, .end = k_end },
    .family = k_family,
    .accesses = { .anchor = writer, .readers = { reader } } } };
  auto pieces = partition_buffer_regions(original, { .begin = k_quarter, .end = k_three_quarters });
  REQUIRE(pieces.has_value());
  REQUIRE(pieces->size() == k_split_count);
  for (auto const &piece : *pieces) {
    REQUIRE(piece.accesses.anchor == writer);
    REQUIRE(piece.accesses.readers == std::vector{ reader });
    REQUIRE(piece.family == k_family);
  }
  auto differing = *pieces;
  differing.at(k_second_step).accesses.readers.clear();
  merge_buffer_regions(differing);
  REQUIRE(differing.size() == k_split_count);
  merge_buffer_regions(*pieces);
  REQUIRE(pieces->size() == k_one_region);
  REQUIRE(pieces->front().range == original.front().range);
}

TEST_CASE("partitioning fills finite holes and rejects open uncovered tails", "[vkexec][ranges][resource-helpers]")
{
  using vkexec::detail::buffer_range;
  using vkexec::detail::buffer_region_state;
  using vkexec::detail::partition_buffer_regions;

  std::vector const original{ buffer_region_state{
    .range = { .begin = k_quarter, .end = k_middle }, .family = k_family, .accesses = {} } };
  auto const open_extension = partition_buffer_regions(original, { .begin = k_begin, .end = std::nullopt });
  REQUIRE_FALSE(open_extension.has_value());

  auto const finite = partition_buffer_regions(original, { .begin = k_begin, .end = k_end });
  REQUIRE(finite.has_value());
  REQUIRE(finite->size() == k_split_count);
  REQUIRE(finite->front().range == buffer_range{ .begin = k_begin, .end = k_quarter });
  REQUIRE(finite->at(k_second_step).range == original.front().range);
  REQUIRE(finite->back().range == buffer_range{ .begin = k_middle, .end = k_end });

  auto const first_open = partition_buffer_regions({}, { .begin = k_begin, .end = std::nullopt });
  REQUIRE(first_open.has_value());
  REQUIRE(first_open->size() == k_one_region);
  REQUIRE(first_open->front().range == buffer_range{ .begin = k_begin, .end = std::nullopt });
}

TEST_CASE("partitioning rejects finite cuts of open tracked buffer ranges", "[vkexec][ranges][resource-helpers]")
{
  using vkexec::detail::buffer_region_state;
  using vkexec::detail::partition_buffer_regions;

  std::vector const original{ buffer_region_state{
    .range = { .begin = k_begin, .end = std::nullopt }, .family = k_family, .accesses = {} } };
  auto const pieces = partition_buffer_regions(original, { .begin = k_begin, .end = k_end });
  REQUIRE_FALSE(pieces.has_value());
}

TEST_CASE("barrier coalescing preserves synchronization scopes and transfer pairs",
  "[vkexec][ranges][barrier-coalescing]")
{
  using vkexec::detail::merge_barriers;

  vkexec::buffer_barrier_params first{ .offset = k_begin,
    .size = k_quarter,
    .src_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .dst_stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
    .src_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    .dst_access = VK_ACCESS_2_TRANSFER_READ_BIT };
  auto second = first;
  second.offset = k_quarter;
  auto merged = first;
  REQUIRE(merge_barriers(merged, second));
  REQUIRE(merged.size == k_middle);
  second.dst_access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  REQUIRE_FALSE(merge_barriers(first, second));
  second = first;
  second.offset = k_quarter;
  constexpr std::uint32_t k_other_family = 1;
  first.src_queue_family = second.src_queue_family = k_family;
  first.dst_queue_family = second.dst_queue_family = k_other_family;
  REQUIRE_FALSE(merge_barriers(first, second));
}

TEST_CASE("image barrier coalescing requires a rectangular union and equal old layouts",
  "[vkexec][ranges][barrier-coalescing]")
{
  using vkexec::detail::merge_barriers;

  constexpr std::uint32_t k_base_mip = 0;
  constexpr std::uint32_t k_next_mip = 1;
  constexpr std::uint32_t k_mip_count = 2;
  constexpr std::uint32_t k_next_layer = 1;
  vkexec::image_barrier_params first{};
  first.old_layout = VK_IMAGE_LAYOUT_GENERAL;
  first.new_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  auto second = first;
  second.range.baseMipLevel = k_next_mip;
  auto merged = first;
  REQUIRE(merge_barriers(merged, second));
  REQUIRE(merged.range.baseMipLevel == k_base_mip);
  REQUIRE(merged.range.levelCount == k_mip_count);
  second.old_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  REQUIRE_FALSE(merge_barriers(first, second));
  second.old_layout = first.old_layout;
  second.range.baseArrayLayer = k_next_layer;
  REQUIRE_FALSE(merge_barriers(first, second));
}

// NOLINTEND(readability-function-cognitive-complexity)
