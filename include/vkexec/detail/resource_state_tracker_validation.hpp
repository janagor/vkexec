#ifndef VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_VALIDATION_HPP
#define VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_VALIDATION_HPP

#include <vkexec/detail/resource_state_tracker_helpers.hpp>

namespace vkexec::detail {

  template<class Access>
  [[nodiscard]] inline auto frontier_handoff_queue(access_frontier<Access> const &frontier) -> VkQueue
  {
    if (!frontier.readers.empty()) { return frontier.readers.back().queue; }
    if (frontier.anchor) { return frontier.anchor->queue; }
    return VK_NULL_HANDLE;
  }

  [[nodiscard]] inline auto clamp_image_region(image_region &region, image_extent const &extent) -> status
  {
    if (region.mips.begin >= extent.mip_levels || (region.mips.end && *region.mips.end > extent.mip_levels)
        || region.layers.begin >= extent.array_layers
        || (region.layers.end && *region.layers.end > extent.array_layers)) {
      return fail(errc::invalid_argument, "graph image range exceeds registered extent");
    }
    if (!region.mips.end) { region.mips.end = extent.mip_levels; }
    if (!region.layers.end) { region.layers.end = extent.array_layers; }
    return {};
  }

  [[nodiscard]] inline auto clamp_buffer_range(buffer_range &range, VkDeviceSize size) -> status
  {
    if (range.begin >= size || (range.end && *range.end > size)) {
      return fail(errc::invalid_argument, "graph buffer range exceeds registered extent");
    }
    if (!range.end) { range.end = size; }
    return {};
  }

  [[nodiscard]] inline auto reject_mixed_aspect_declarations(std::vector<image_region_state> const &states,
    image_region const &region) -> status
  {
    if (std::ranges::any_of(states, [&region](image_region_state const &state) -> bool {
          return (state.region.aspects & region.aspects) != 0 && state.region.aspects != region.aspects;
        })) {
      return fail(errc::unsupported, "cannot mix coupled and separate depth/stencil declarations for one image");
    }
    return {};
  }

  [[nodiscard]] inline auto validate_image_use_conflicts(std::vector<image_region_state> const &regions,
    image_region const &region,
    image_use const &declaration,
    std::size_t step,
    std::uint32_t family,
    VkQueue queue,
    bool exclusive,
    usage_scope next,
    usage_scope after,
    bool render_pass_transition) -> status
  {
    for (auto const &state : regions) {
      if (!overlaps(state.region, region)) { continue; }
      auto const &frontier = state.accesses;
      if (!frontier.anchor && frontier.readers.empty()) {
        if (declaration.access != resource_access::write && declaration.initial_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
          return fail(errc::invalid_argument, "first graph image read requires an initial layout");
        }
        continue;
      }
      bool const compatible_read = declaration.access == resource_access::read && (!exclusive || state.family == family)
                                   && state.layout == next.layout && after.layout == next.layout
                                   && !render_pass_transition;
      auto const conflicts = [step, queue, compatible_read](tracked_image const &access) -> bool {
        return access.step == step
               && (!compatible_read || access.access != resource_access::read || access.queue != queue);
      };
      if ((frontier.anchor && frontier.anchor->step == step) || std::ranges::any_of(frontier.readers, conflicts)) {
        return fail(errc::invalid_argument, "overlapping image uses in one pass require one combined declaration");
      }
    }
    return {};
  }

  [[nodiscard]] inline auto validate_buffer_use_conflicts(std::vector<buffer_region_state> const &regions,
    buffer_range range,
    buffer_use const &declaration,
    std::size_t step,
    std::uint32_t family,
    VkQueue queue) -> status
  {
    for (auto const &region : regions) {
      if (!overlaps(region.range, range)) { continue; }
      auto const conflicts = [declaration, step, family, queue](tracked_buffer const &access) -> bool {
        return access.step == step
               && (access.access != resource_access::read || declaration.access != resource_access::read
                   || access.family != family || access.queue != queue);
      };
      if ((region.accesses.anchor && region.accesses.anchor->step == step)
          || std::ranges::any_of(region.accesses.readers, conflicts)) {
        return fail(errc::invalid_argument, "overlapping buffer uses in one pass require one combined declaration");
      }
    }
    return {};
  }

  template<class Access>
  [[nodiscard]] inline auto source_scope(access_frontier<Access> const &frontier, VkQueue queue) -> usage_scope
  {
    usage_scope combined{};
    auto const include = [&combined, queue](Access const &access) -> void {
      if (access.queue == VK_NULL_HANDLE || queue == VK_NULL_HANDLE || access.queue == queue) {
        combined.stage |= access.scope.stage;
        combined.access |= access.scope.access;
      }
    };
    if (frontier.readers.empty()) {
      if (frontier.anchor) { include(*frontier.anchor); }
    } else {
      for (auto const &reader : frontier.readers) { include(reader); }
    }
    return combined;
  }

} // namespace vkexec::detail

#endif// VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_VALIDATION_HPP
