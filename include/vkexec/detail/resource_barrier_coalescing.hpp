#ifndef VKEXEC_DETAIL_RESOURCE_BARRIER_COALESCING_HPP
#define VKEXEC_DETAIL_RESOURCE_BARRIER_COALESCING_HPP

#include <vkexec/barrier_params.hpp>
#include <vkexec/detail/resource_ranges.hpp>

#include <cstddef>
#include <vector>

namespace vkexec::detail {

template<class Barrier>
[[nodiscard]] constexpr auto same_barrier_scope(Barrier const &lhs, Barrier const &rhs) noexcept -> bool
{
  return lhs.src_stage == rhs.src_stage && lhs.dst_stage == rhs.dst_stage && lhs.src_access == rhs.src_access
         && lhs.dst_access == rhs.dst_access && lhs.src_queue_family == rhs.src_queue_family
         && lhs.dst_queue_family == rhs.dst_queue_family;
}

[[nodiscard]] inline auto merge_barriers(buffer_barrier_params &lhs, buffer_barrier_params const &rhs) -> bool
{
  // Ownership transfers retain the exact range shared by their release/acquire pair.
  if (lhs.buffer != rhs.buffer || lhs.src_queue_family != lhs.dst_queue_family || !same_barrier_scope(lhs, rhs)) {
    return false;
  }
  auto const left = normalize(buffer_use{ .offset = lhs.offset, .size = lhs.size });
  auto const right = normalize(buffer_use{ .offset = rhs.offset, .size = rhs.size });
  if (!left || !right || !adjacent(*left, *right)) { return false; }
  auto const combined = interval_union(*left, *right);
  lhs.offset = combined.begin;
  lhs.size = region_size(combined);
  return true;
}

[[nodiscard]] inline auto merge_barriers(image_barrier_params &lhs, image_barrier_params const &rhs) -> bool
{
  if (lhs.image != rhs.image || lhs.old_layout != rhs.old_layout || lhs.new_layout != rhs.new_layout
      || lhs.src_queue_family != lhs.dst_queue_family || !same_barrier_scope(lhs, rhs)) {
    return false;
  }
  auto const left = normalize(lhs.range);
  auto const right = normalize(rhs.range);
  if (!left || !right) { return false; }
  auto const combined = merge_compatible(*left, *right);
  if (!combined) { return false; }
  lhs.range = subresource_range(*combined);
  return true;
}

template<class Barrier> auto coalesce_barriers(std::vector<Barrier> &barriers) -> void
{
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t left{}; left < barriers.size(); ++left) {
      for (std::size_t right = left + 1; right < barriers.size();) {
        if (merge_barriers(barriers.at(left), barriers.at(right))) {
          barriers.erase(barriers.begin() + static_cast<std::ptrdiff_t>(right));
          changed = true;
        } else {
          ++right;
        }
      }
    }
  }
}

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_RESOURCE_BARRIER_COALESCING_HPP
