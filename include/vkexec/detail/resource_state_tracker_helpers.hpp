#ifndef VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_HELPERS_HPP
#define VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_HELPERS_HPP

#include <vkexec/barrier_params.hpp>
#include <vkexec/config.hpp>
#include <vkexec/detail/resource_barrier_coalescing.hpp>
#include <vkexec/detail/resource_ranges.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/result.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#ifdef VKEXEC_ENABLE_ALLOCATION_FAULT_INJECTION
#ifndef VKEXEC_HAS_EXCEPTIONS
#error "VKEXEC_ENABLE_ALLOCATION_FAULT_INJECTION requires <vkexec/config.hpp> (VKEXEC_HAS_EXCEPTIONS)"
#endif
#if VKEXEC_HAS_EXCEPTIONS
#include <new>
#endif
#endif
#include <optional>
#include <ranges>
#include <utility>
#include <vector>

namespace vkexec::detail {

struct resource_sync_point
{
  std::vector<image_barrier_params> images_before;
  std::vector<buffer_barrier_params> buffers_before;
  std::vector<image_barrier_params> images_after;
  std::vector<buffer_barrier_params> buffers_after;
};

struct resource_sync_plan
{
  std::vector<resource_sync_point> steps;
  std::vector<std::pair<std::size_t, std::size_t>> dependencies;
  struct image_access
  {
    VkImage image{ VK_NULL_HANDLE };
    image_usage usage{};
    VkImageLayout final_layout{ VK_IMAGE_LAYOUT_UNDEFINED };
    std::size_t step{};
    VkImageSubresourceRange range{};
  };
  std::vector<image_access> image_uses;
};

struct usage_scope
{
  VkPipelineStageFlags2 stage{ VK_PIPELINE_STAGE_2_NONE };
  VkAccessFlags2 access{ VK_ACCESS_2_NONE };
  VkImageLayout layout{ VK_IMAGE_LAYOUT_UNDEFINED };

  [[nodiscard]] auto operator==(usage_scope const &) const -> bool = default;
};

[[nodiscard]] constexpr auto
  access_mask(resource_access access, VkAccessFlags2 read_mask, VkAccessFlags2 write_mask) noexcept -> VkAccessFlags2
{
  switch (access) {
  case resource_access::read:
    return read_mask;
  case resource_access::write:
    return write_mask;
  case resource_access::read_write:
    return read_mask | write_mask;
  }
  return VK_ACCESS_2_NONE;
}

[[nodiscard]] constexpr auto valid(image_use const &use) noexcept -> bool
{
  switch (use.usage) {
  case image_usage::sampled_compute:
  case image_usage::sampled_fragment:
  case image_usage::sampled_graphics:
  case image_usage::transfer_source:
    return use.access == resource_access::read;
  case image_usage::transfer_destination:
    return use.access == resource_access::write;
  case image_usage::storage_compute:
  case image_usage::storage_graphics:
  case image_usage::color_attachment:
  case image_usage::depth_attachment:
    return true;
  }
  return false;
}

[[nodiscard]] constexpr auto valid(buffer_use const &use) noexcept -> bool
{
  switch (use.usage) {
  case buffer_usage::indirect:
  case buffer_usage::vertex:
  case buffer_usage::index:
  case buffer_usage::uniform_compute:
  case buffer_usage::uniform_vertex:
  case buffer_usage::uniform_fragment:
  case buffer_usage::transfer_source:
    return use.access == resource_access::read;
  case buffer_usage::transfer_destination:
    return use.access == resource_access::write;
  case buffer_usage::storage_compute:
  case buffer_usage::storage_graphics:
    return true;
  }
  return false;
}

[[nodiscard]] constexpr auto default_image_scope(image_use use) noexcept -> usage_scope
{
  switch (use.usage) {
  case image_usage::sampled_compute:
    return { .stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
      .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
  case image_usage::sampled_fragment:
    return { .stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
      .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
  case image_usage::sampled_graphics:
    return { .stage = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
      .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
      .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
  case image_usage::storage_compute:
    return { .stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .access = access_mask(use.access, VK_ACCESS_2_SHADER_STORAGE_READ_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
      .layout = VK_IMAGE_LAYOUT_GENERAL };
  case image_usage::storage_graphics:
    return { .stage = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
      .access = access_mask(use.access, VK_ACCESS_2_SHADER_STORAGE_READ_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
      .layout = VK_IMAGE_LAYOUT_GENERAL };
  case image_usage::color_attachment:
    return { .stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      .access = access_mask(use.access, VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT),
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
  case image_usage::depth_attachment:
    return { .stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
      .access = access_mask(
        use.access, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
      .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
  case image_usage::transfer_source:
    return { .stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
      .access = VK_ACCESS_2_TRANSFER_READ_BIT,
      .layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL };
  case image_usage::transfer_destination:
    return { .stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
      .access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL };
  }
  return {};
}

[[nodiscard]] constexpr auto image_scope(image_use use) noexcept -> usage_scope
{
  auto scope = default_image_scope(use);
  if (use.layout != VK_IMAGE_LAYOUT_UNDEFINED) { scope.layout = use.layout; }
  return scope;
}

[[nodiscard]] constexpr auto buffer_scope(buffer_use use) noexcept -> usage_scope
{
  switch (use.usage) {
  case buffer_usage::indirect:
    return { .stage = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, .access = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT };
  case buffer_usage::vertex:
    return { .stage = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT, .access = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT };
  case buffer_usage::index:
    return { .stage = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT, .access = VK_ACCESS_2_INDEX_READ_BIT };
  case buffer_usage::uniform_compute:
    return { .stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, .access = VK_ACCESS_2_UNIFORM_READ_BIT };
  case buffer_usage::uniform_vertex:
    return { .stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, .access = VK_ACCESS_2_UNIFORM_READ_BIT };
  case buffer_usage::uniform_fragment:
    return { .stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, .access = VK_ACCESS_2_UNIFORM_READ_BIT };
  case buffer_usage::storage_compute:
    return { .stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .access = access_mask(use.access, VK_ACCESS_2_SHADER_STORAGE_READ_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) };
  case buffer_usage::storage_graphics:
    return { .stage = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
      .access = access_mask(use.access, VK_ACCESS_2_SHADER_STORAGE_READ_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) };
  case buffer_usage::transfer_source:
    return { .stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_READ_BIT };
  case buffer_usage::transfer_destination:
    return { .stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_WRITE_BIT };
  }
  return {};
}

struct tracked_access
{
  resource_access access{};
  usage_scope scope;
  std::size_t step{};
  std::uint32_t family{};
  VkQueue queue{ VK_NULL_HANDLE };

  [[nodiscard]] auto operator==(tracked_access const &) const -> bool = default;
};

using tracked_image = tracked_access;
using tracked_buffer = tracked_access;

template<class Access> struct access_frontier
{
  std::optional<Access> anchor;
  std::vector<Access> readers;

  [[nodiscard]] auto operator==(access_frontier const &) const -> bool = default;
};

struct image_region_state
{
  image_region region;
  VkImageLayout layout{ VK_IMAGE_LAYOUT_UNDEFINED };
  std::uint32_t family{ VK_QUEUE_FAMILY_IGNORED };
  access_frontier<tracked_image> accesses;
};

struct buffer_region_state
{
  buffer_range range;
  std::uint32_t family{ VK_QUEUE_FAMILY_IGNORED };
  access_frontier<tracked_buffer> accesses;
};

struct tracked_image_resource
{
  VkImage image{ VK_NULL_HANDLE };
  std::vector<image_region_state> regions;
  std::optional<image_extent> extent;
};

struct tracked_buffer_resource
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  std::vector<buffer_region_state> regions;
  std::optional<VkDeviceSize> size;
  VkSharingMode sharing_mode{ VK_SHARING_MODE_EXCLUSIVE };
};

struct resource_region_counts
{
  std::size_t buffers{};
  std::size_t images{};
};

//! Test-only fault seam. Active only when `VKEXEC_ENABLE_ALLOCATION_FAULT_INJECTION` is defined
//! for a whole test executable (see `vkexec_core_tests`) so inline definitions stay ODR-consistent.
struct resource_allocation_fault
{
#if defined(VKEXEC_ENABLE_ALLOCATION_FAULT_INJECTION) && VKEXEC_HAS_EXCEPTIONS
  static constexpr bool k_enabled = true;
  static inline thread_local int remaining{};

  static auto arm(int count) noexcept -> void { remaining = count; }

  static auto disarm() noexcept -> void { remaining = 0; }
#else
  static constexpr bool k_enabled = false;
  static auto arm(int /*unused*/) noexcept -> void {}

  static auto disarm() noexcept -> void {}
#endif

  //! Throws on the Nth armed checkpoint (`arm(1)` → first call, `arm(2)` → second, …).
  static auto maybe_throw() noexcept(!k_enabled) -> void
  {
#if defined(VKEXEC_ENABLE_ALLOCATION_FAULT_INJECTION) && VKEXEC_HAS_EXCEPTIONS
    if (remaining > 0) {
      --remaining;
      if (remaining == 0) { throw std::bad_alloc{}; }
    }
#endif
  }
};

//! Copies frontiers into every child and creates empty frontiers for uncovered bytes.
//! Rejects finite cuts of open ranges and open uncovered gaps that would retain an unproven symbolic tail.
[[nodiscard]] inline auto partition_buffer_regions(std::vector<buffer_region_state> const &regions, buffer_range range)
  -> result<std::vector<buffer_region_state>>
{
  std::vector<buffer_region_state> partitioned;
  std::optional<VkDeviceSize> cursor = range.begin;
  bool const extending_tracked = !regions.empty();
  for (auto const &region : regions) {
    auto const common = intersection(region.range, range);
    if (!common) {
      partitioned.push_back(region);
      continue;
    }
    if (cursor && *cursor < common->begin) {
      buffer_range const gap{ .begin = *cursor, .end = common->begin };
      if (ambiguous_open_uncovered(gap, extending_tracked)) {
        return fail(errc::invalid_argument,
          "graph buffer use extends a finite tracked range into an open suffix; register buffer_extent to resolve the "
          "bound");
      }
      partitioned.push_back(buffer_region_state{ .range = gap, .family = VK_QUEUE_FAMILY_IGNORED, .accesses = {} });
    }
    if (auto const pieces = split(region.range, common->begin)) {
      auto left = region;
      left.range = pieces->first;
      partitioned.push_back(std::move(left));
    }
    auto middle = region;
    middle.range = *common;
    partitioned.push_back(std::move(middle));
    if (common->end) {
      if (ambiguous_open_remainder(region.range, *common->end)) {
        return fail(errc::invalid_argument,
          "graph buffer use splits an open range at a finite end; register buffer_extent to resolve the bound");
      }
      if (auto const pieces = split(region.range, *common->end)) {
        auto right = region;
        right.range = pieces->second;
        partitioned.push_back(std::move(right));
      }
    }
    cursor = common->end;
  }
  if (cursor && (!range.end || *cursor < *range.end)) {
    buffer_range const gap{ .begin = *cursor, .end = range.end };
    if (ambiguous_open_uncovered(gap, extending_tracked)) {
      return fail(errc::invalid_argument,
        "graph buffer use extends a finite tracked range into an open suffix; register buffer_extent to resolve the "
        "bound");
    }
    partitioned.push_back(buffer_region_state{ .range = gap, .family = VK_QUEUE_FAMILY_IGNORED, .accesses = {} });
  }
  std::ranges::sort(
    partitioned, {}, [](buffer_region_state const &region) -> VkDeviceSize { return region.range.begin; });
  return partitioned;
}

inline auto merge_buffer_regions(std::vector<buffer_region_state> &regions) -> void
{
  std::size_t retained{};
  for (std::size_t index{}; index < regions.size(); ++index) {
    auto &region = regions.at(index);
    if (retained != 0) {
      auto &previous = regions.at(retained - 1);
      if (adjacent(previous.range, region.range) && previous.family == region.family
          && previous.accesses.anchor == region.accesses.anchor
          && previous.accesses.readers == region.accesses.readers) {
        previous.range.end = region.range.end;
        continue;
      }
    }
    if (retained != index) { regions.at(retained) = std::move(region); }
    ++retained;
  }
  regions.resize(retained);
}

[[nodiscard]] inline auto partition_image_regions(std::vector<image_region_state> const &regions, image_region region)
  -> result<std::vector<image_region_state>>
{
  std::vector<image_region_state> partitioned;
  std::vector<image_region> uncovered{ region };
  for (auto const &state : regions) {
    auto const common = intersection(state.region, region);
    if (!common) {
      partitioned.push_back(state);
      continue;
    }
    if (common->mips.end && ambiguous_open_remainder(state.region.mips, *common->mips.end)) {
      return fail(errc::invalid_argument,
        "graph image use splits an open mip range at a finite end; register image_extent to resolve the bound");
    }
    if (common->layers.end && ambiguous_open_remainder(state.region.layers, *common->layers.end)) {
      return fail(errc::invalid_argument,
        "graph image use splits an open layer range at a finite end; register image_extent to resolve the bound");
    }
    for (auto const &remainder : subtract(state.region, *common)) {
      auto piece = state;
      piece.region = remainder;
      partitioned.push_back(std::move(piece));
    }
    auto middle = state;
    middle.region = *common;
    partitioned.push_back(std::move(middle));
    std::vector<image_region> remaining;
    for (auto const &gap : uncovered) { std::ranges::copy(subtract(gap, *common), std::back_inserter(remaining)); }
    uncovered = std::move(remaining);
  }
  for (auto const &gap : uncovered) {
    if (std::ranges::any_of(regions,
          [&gap](image_region_state const &state) -> bool { return ambiguous_open_uncovered(gap, state.region); })) {
      return fail(errc::invalid_argument,
        "graph image use extends a finite tracked range into an open suffix; register image_extent to resolve the "
        "bound");
    }
    partitioned.push_back(image_region_state{ .region = gap, .accesses = {} });
  }
  return partitioned;
}

inline auto merge_image_regions(std::vector<image_region_state> &regions) -> void
{
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t left{}; left < regions.size(); ++left) {
      for (std::size_t right = left + 1; right < regions.size();) {
        auto &lhs = regions.at(left);
        auto const &rhs = regions.at(right);
        auto const combined = merge_compatible(lhs.region, rhs.region);
        if (combined && lhs.layout == rhs.layout && lhs.family == rhs.family && lhs.accesses == rhs.accesses) {
          lhs.region = *combined;
          regions.erase(regions.begin() + static_cast<std::ptrdiff_t>(right));
          changed = true;
        } else {
          ++right;
        }
      }
    }
  }
}

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_HELPERS_HPP
