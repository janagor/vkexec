#ifndef VKEXEC_DETAIL_RESOURCE_RANGES_HPP
#define VKEXEC_DETAIL_RESOURCE_RANGES_HPP

#include <vkexec/resource_use.hpp>
#include <vkexec/result.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace vkexec::detail {

//! Half-open interval; an absent end denotes the end of the resource dimension.
template<class Index> struct resource_interval
{
  Index begin{};
  std::optional<Index> end;

  [[nodiscard]] auto operator==(resource_interval const &) const -> bool = default;
};

using buffer_range = resource_interval<VkDeviceSize>;
using subresource_interval = resource_interval<std::uint32_t>;

[[nodiscard]] inline auto normalize(buffer_use const &use) -> result<buffer_range>
{
  if (use.size == 0 || use.offset == std::numeric_limits<VkDeviceSize>::max()
      || (use.size != VK_WHOLE_SIZE && use.size > std::numeric_limits<VkDeviceSize>::max() - use.offset)) {
    return fail(errc::invalid_argument, "invalid graph buffer range");
  }
  return buffer_range{ .begin = use.offset,
    .end = use.size == VK_WHOLE_SIZE ? std::nullopt : std::optional{ use.offset + use.size } };
}

template<class Index>
[[nodiscard]] constexpr auto overlaps(resource_interval<Index> lhs, resource_interval<Index> rhs) noexcept -> bool
{ return (!lhs.end || rhs.begin < *lhs.end) && (!rhs.end || lhs.begin < *rhs.end); }

template<class Index>
[[nodiscard]] constexpr auto intersection_end(std::optional<Index> lhs, std::optional<Index> rhs) noexcept
  -> std::optional<Index>
{
  if (!lhs) { return rhs; }
  if (!rhs) { return lhs; }
  return std::min(*lhs, *rhs);
}

template<class Index>
[[nodiscard]] constexpr auto intersection(resource_interval<Index> lhs, resource_interval<Index> rhs) noexcept
  -> std::optional<resource_interval<Index>>
{
  if (!overlaps(lhs, rhs)) { return std::nullopt; }
  return resource_interval<Index>{ .begin = std::max(lhs.begin, rhs.begin), .end = intersection_end(lhs.end, rhs.end) };
}

template<class Index>
[[nodiscard]] constexpr auto contains(resource_interval<Index> outer, resource_interval<Index> inner) noexcept -> bool
{ return outer.begin <= inner.begin && (!outer.end || (inner.end && *inner.end <= *outer.end)); }

template<class Index>
[[nodiscard]] constexpr auto split(resource_interval<Index> range, Index position) noexcept
  -> std::optional<std::pair<resource_interval<Index>, resource_interval<Index>>>
{
  if (position <= range.begin || (range.end && position >= *range.end)) { return std::nullopt; }
  return std::pair{ resource_interval<Index>{ .begin = range.begin, .end = position },
    resource_interval<Index>{ .begin = position, .end = range.end } };
}

//! A finite cut of an open interval leaves a symbolic remainder whose nonempty length is unprovable.
template<class Index>
[[nodiscard]] constexpr auto ambiguous_open_remainder(resource_interval<Index> range, Index cut) noexcept -> bool
{ return !range.end && cut > range.begin; }

//! An open uncovered gap past already-tracked coverage cannot prove a nonempty symbolic suffix.
template<class Index>
[[nodiscard]] constexpr auto ambiguous_open_uncovered(resource_interval<Index> gap, bool extending_tracked) noexcept
  -> bool
{ return extending_tracked && !gap.end; }

template<class Index>
[[nodiscard]] constexpr auto adjacent(resource_interval<Index> lhs, resource_interval<Index> rhs) noexcept -> bool
{ return (lhs.end && *lhs.end == rhs.begin) || (rhs.end && *rhs.end == lhs.begin); }

[[nodiscard]] constexpr auto region_size(buffer_range range) noexcept -> VkDeviceSize
{ return range.end ? *range.end - range.begin : VK_WHOLE_SIZE; }

//! Depth and stencil may form one indivisible aspect group for coupled layouts.
struct image_region
{
  VkImageAspectFlags aspects{};
  subresource_interval mips;
  subresource_interval layers;

  [[nodiscard]] auto operator==(image_region const &) const -> bool = default;
};

//! Open mip or layer ends are ambiguous only against tracked coverage that shares aspects and the other dimension.
[[nodiscard]] constexpr auto ambiguous_open_uncovered(image_region const &gap, image_region const &tracked) noexcept
  -> bool
{
  if ((gap.aspects & tracked.aspects) == 0) { return false; }
  if (!gap.mips.end && overlaps(gap.layers, tracked.layers)) { return true; }
  if (!gap.layers.end && overlaps(gap.mips, tracked.mips)) { return true; }
  return false;
}

[[nodiscard]] inline auto normalize_subresources(std::uint32_t begin, std::uint32_t count, std::uint32_t remaining)
  -> result<subresource_interval>
{
  if (count == 0 || begin == std::numeric_limits<std::uint32_t>::max()
      || (count != remaining && count > std::numeric_limits<std::uint32_t>::max() - begin)) {
    return fail(errc::invalid_argument, "invalid graph image subresource interval");
  }
  return subresource_interval{ .begin = begin,
    .end = count == remaining ? std::nullopt : std::optional{ begin + count } };
}

[[nodiscard]] inline auto normalize(VkImageSubresourceRange range) -> result<image_region>
{
  constexpr VkImageAspectFlags k_depth_stencil = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  if (range.aspectMask == 0) { return fail(errc::invalid_argument, "empty graph image aspect mask"); }
  if (range.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT && (range.aspectMask & ~k_depth_stencil) != 0) {
    return fail(
      errc::unsupported, "graph images support color or depth/stencil aspects; plane aspects are unsupported");
  }
  VKEXEC_TRY_ASSIGN(mips, normalize_subresources(range.baseMipLevel, range.levelCount, VK_REMAINING_MIP_LEVELS));
  VKEXEC_TRY_ASSIGN(layers, normalize_subresources(range.baseArrayLayer, range.layerCount, VK_REMAINING_ARRAY_LAYERS));
  return image_region{ .aspects = range.aspectMask, .mips = mips, .layers = layers };
}

[[nodiscard]] constexpr auto overlaps(image_region const &lhs, image_region const &rhs) noexcept -> bool
{ return (lhs.aspects & rhs.aspects) != 0 && overlaps(lhs.mips, rhs.mips) && overlaps(lhs.layers, rhs.layers); }

[[nodiscard]] constexpr auto intersection(image_region const &lhs, image_region const &rhs) noexcept
  -> std::optional<image_region>
{
  auto const mips = intersection(lhs.mips, rhs.mips);
  auto const layers = intersection(lhs.layers, rhs.layers);
  if ((lhs.aspects & rhs.aspects) == 0 || !mips || !layers) { return std::nullopt; }
  return image_region{ .aspects = lhs.aspects & rhs.aspects, .mips = *mips, .layers = *layers };
}

[[nodiscard]] constexpr auto subresource_range(image_region const &region) noexcept -> VkImageSubresourceRange
{
  return { .aspectMask = region.aspects,
    .baseMipLevel = region.mips.begin,
    .levelCount = region.mips.end ? *region.mips.end - region.mips.begin : VK_REMAINING_MIP_LEVELS,
    .baseArrayLayer = region.layers.begin,
    .layerCount = region.layers.end ? *region.layers.end - region.layers.begin : VK_REMAINING_ARRAY_LAYERS };
}

//! Subtracts a rectangle, preserving any aspects outside the intersection.
[[nodiscard]] inline auto subtract(image_region const &region, image_region const &removed) -> std::vector<image_region>
{
  auto const common = intersection(region, removed);
  if (!common) { return { region }; }
  std::vector<image_region> pieces;
  if (auto const aspects = region.aspects & ~common->aspects; aspects != 0) {
    pieces.push_back(image_region{ .aspects = aspects, .mips = region.mips, .layers = region.layers });
  }
  auto const append = [&](subresource_interval mips, subresource_interval layers) -> void {
    pieces.push_back(image_region{ .aspects = common->aspects, .mips = mips, .layers = layers });
  };
  if (region.mips.begin < common->mips.begin) {
    append({ .begin = region.mips.begin, .end = common->mips.begin }, region.layers);
  }
  if (common->mips.end && common->mips.end != region.mips.end) {
    append({ .begin = *common->mips.end, .end = region.mips.end }, region.layers);
  }
  if (region.layers.begin < common->layers.begin) {
    append(common->mips, { .begin = region.layers.begin, .end = common->layers.begin });
  }
  if (common->layers.end && common->layers.end != region.layers.end) {
    append(common->mips, { .begin = *common->layers.end, .end = region.layers.end });
  }
  return pieces;
}

template<class Index>
[[nodiscard]] constexpr auto interval_union(resource_interval<Index> lhs, resource_interval<Index> rhs) noexcept
  -> resource_interval<Index>
{
  return { .begin = std::min(lhs.begin, rhs.begin),
    .end = !lhs.end || !rhs.end ? std::nullopt : std::optional{ std::max(*lhs.end, *rhs.end) } };
}

//! Only a rectangular union with the same aspect group is representable here.
[[nodiscard]] inline auto merge_compatible(image_region const &lhs, image_region const &rhs) noexcept
  -> std::optional<image_region>
{
  if (lhs.aspects != rhs.aspects) { return std::nullopt; }
  if (lhs.mips == rhs.mips && adjacent(lhs.layers, rhs.layers)) {
    return image_region{ .aspects = lhs.aspects, .mips = lhs.mips, .layers = interval_union(lhs.layers, rhs.layers) };
  }
  if (lhs.layers == rhs.layers && adjacent(lhs.mips, rhs.mips)) {
    return image_region{ .aspects = lhs.aspects, .mips = interval_union(lhs.mips, rhs.mips), .layers = lhs.layers };
  }
  return std::nullopt;
}

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_RESOURCE_RANGES_HPP
