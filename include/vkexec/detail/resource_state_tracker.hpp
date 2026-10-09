#ifndef VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_HPP
#define VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_HPP

#include <vkexec/barrier_params.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/result.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
  };
  std::vector<image_access> image_uses;
};

struct usage_scope
{
  VkPipelineStageFlags2 stage{ VK_PIPELINE_STAGE_2_NONE };
  VkAccessFlags2 access{ VK_ACCESS_2_NONE };
  VkImageLayout layout{ VK_IMAGE_LAYOUT_UNDEFINED };
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

struct tracked_image
{
  image_use use;
  usage_scope scope;
  std::size_t step{};
  std::uint32_t family{};
  VkQueue queue{ VK_NULL_HANDLE };
};

struct tracked_buffer
{
  buffer_use use;
  usage_scope scope;
  std::size_t step{};
  std::uint32_t family{};
  VkQueue queue{ VK_NULL_HANDLE };
};

template<class Access> struct access_frontier
{
  std::optional<Access> anchor;
  std::vector<Access> readers;
};

struct tracked_image_state
{
  VkImage image{ VK_NULL_HANDLE };
  VkImageSubresourceRange range{};
  VkImageLayout layout{ VK_IMAGE_LAYOUT_UNDEFINED };
  std::uint32_t family{ VK_QUEUE_FAMILY_IGNORED };
  access_frontier<tracked_image> accesses;
};

struct tracked_buffer_state
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  VkDeviceSize offset{};
  VkDeviceSize size{};
  std::uint32_t family{ VK_QUEUE_FAMILY_IGNORED };
  access_frontier<tracked_buffer> accesses;
};

class resource_state_tracker
{
  resource_sync_plan plan_;
  std::vector<tracked_image_state> images_;
  std::vector<tracked_buffer_state> buffers_;

  [[nodiscard]] static auto same_range(VkImageSubresourceRange lhs, VkImageSubresourceRange rhs) noexcept -> bool
  {
    return lhs.aspectMask == rhs.aspectMask && lhs.baseMipLevel == rhs.baseMipLevel && lhs.levelCount == rhs.levelCount
           && lhs.baseArrayLayer == rhs.baseArrayLayer && lhs.layerCount == rhs.layerCount;
  }

  auto add_dependency(std::size_t source, std::size_t destination) -> void
  {
    auto const edge = std::pair{ source, destination };
    if (source != destination && std::ranges::find(plan_.dependencies, edge) == plan_.dependencies.end()) {
      plan_.dependencies.push_back(edge);
    }
  }

  template<class Access> auto depend_on_frontier(access_frontier<Access> const &frontier, std::size_t step) -> void
  {
    if (frontier.readers.empty()) {
      if (frontier.anchor) { add_dependency(frontier.anchor->step, step); }
    } else {
      for (auto const &reader : frontier.readers) { add_dependency(reader.step, step); }
    }
  }

  template<class Access>
  [[nodiscard]] auto collapse_frontier_for_transfer(access_frontier<Access> &frontier) -> std::optional<Access>
  {
    if (!frontier.readers.empty()) {
      Access handoff = frontier.readers.back();
      for (auto const &reader : frontier.readers) { add_dependency(reader.step, handoff.step); }
      frontier.anchor = handoff;
      frontier.readers.clear();
    }
    return frontier.anchor;
  }

  template<class Access>
  [[nodiscard]] static auto source_scope(access_frontier<Access> const &frontier, VkQueue queue) -> usage_scope
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

  [[nodiscard]] auto first_image_use(image_use declaration,
    std::size_t step,
    std::uint32_t family,
    VkQueue queue,
    bool render_pass_transition,
    usage_scope next,
    usage_scope after) -> status
  {
    if (declaration.access != resource_access::write && declaration.initial_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
      return fail(errc::invalid_argument, "first graph image read requires an initial layout");
    }
    if (render_pass_transition) { next.layout = declaration.initial_layout; }
    if (next.layout != VK_IMAGE_LAYOUT_UNDEFINED) {
      plan_.steps.at(step).images_before.push_back(image_barrier_params{ .image = declaration.image,
        .range = declaration.range,
        .old_layout = declaration.initial_layout,
        .new_layout = next.layout,
        .dst_stage = next.stage,
        .dst_access = next.access });
    }
    auto const first_access =
      tracked_image{ .use = declaration, .scope = after, .step = step, .family = family, .queue = queue };
    bool const imported_read = declaration.access == resource_access::read && declaration.initial_layout == next.layout
                               && after.layout == next.layout && !render_pass_transition;
    access_frontier<tracked_image> frontier{};
    if (imported_read) {
      frontier.readers.push_back(first_access);
    } else {
      frontier.anchor = first_access;
    }
    images_.push_back(tracked_image_state{ .image = declaration.image,
      .range = declaration.range,
      .layout = after.layout,
      .family = family,
      .accesses = std::move(frontier) });
    return {};
  }

  auto append_compatible_image_reader(tracked_image_state &state,
    image_use declaration,
    std::size_t step,
    std::uint32_t family,
    VkQueue queue,
    usage_scope next,
    usage_scope after) -> void
  {
    auto &frontier = state.accesses;
    if (frontier.anchor) {
      auto const &anchor = *frontier.anchor;
      add_dependency(anchor.step, step);
      if (anchor.use.access != resource_access::read) {
        auto const source = source_scope(access_frontier<tracked_image>{ .anchor = anchor, .readers = {} }, queue);
        plan_.steps.at(step).images_before.push_back(image_barrier_params{ .image = declaration.image,
          .range = declaration.range,
          .old_layout = state.layout,
          .new_layout = next.layout,
          .src_stage = source.stage,
          .dst_stage = next.stage,
          .src_access = source.access,
          .dst_access = next.access });
      }
    }
    frontier.readers.push_back(
      tracked_image{ .use = declaration, .scope = after, .step = step, .family = family, .queue = queue });
  }

  [[nodiscard]] auto advance_image_state(tracked_image_state &state,
    image_use declaration,
    std::size_t step,
    std::uint32_t family,
    VkQueue queue,
    usage_scope next,
    usage_scope after) -> status
  {
    auto &frontier = state.accesses;
    depend_on_frontier(frontier, step);
    if (state.family != family) {
      auto const handoff = collapse_frontier_for_transfer(frontier);
      if (!handoff) { return fail(errc::invalid_argument, "image transfer requires a previous access"); }
      plan_.steps.at(handoff->step)
        .images_after.push_back(image_barrier_params{ .image = declaration.image,
          .range = declaration.range,
          .old_layout = state.layout,
          .new_layout = next.layout,
          .src_stage = handoff->scope.stage,
          .src_access = handoff->scope.access,
          .src_queue_family = state.family,
          .dst_queue_family = family });
      plan_.steps.at(step).images_before.push_back(image_barrier_params{ .image = declaration.image,
        .range = declaration.range,
        .old_layout = state.layout,
        .new_layout = next.layout,
        .dst_stage = next.stage,
        .dst_access = next.access,
        .src_queue_family = state.family,
        .dst_queue_family = family });
    } else {
      auto const source = source_scope(frontier, queue);
      plan_.steps.at(step).images_before.push_back(image_barrier_params{ .image = declaration.image,
        .range = declaration.range,
        .old_layout = state.layout,
        .new_layout = next.layout,
        .src_stage = source.stage,
        .dst_stage = next.stage,
        .src_access = source.access,
        .dst_access = next.access });
    }
    state.layout = after.layout;
    state.family = family;
    frontier.anchor =
      tracked_image{ .use = declaration, .scope = after, .step = step, .family = family, .queue = queue };
    frontier.readers.clear();
    if (declaration.access == resource_access::read) { frontier.readers.push_back(*frontier.anchor); }
    return {};
  }

  [[nodiscard]] auto
    use_image(image_use declaration, std::size_t step, std::uint32_t family, VkQueue queue, bool render_pass_transition)
      -> status
  {
    if (declaration.image == VK_NULL_HANDLE || declaration.range.aspectMask == 0 || declaration.range.levelCount == 0
        || declaration.range.layerCount == 0) {
      return fail(errc::invalid_argument, "invalid graph image use");
    }
    if (!valid(declaration)) { return fail(errc::invalid_argument, "image usage and access are incompatible"); }
    plan_.image_uses.push_back(resource_sync_plan::image_access{
      .image = declaration.image, .usage = declaration.usage, .final_layout = declaration.final_layout, .step = step });
    usage_scope next = image_scope(declaration);
    usage_scope after = next;
    if (declaration.final_layout != VK_IMAGE_LAYOUT_UNDEFINED) { after.layout = declaration.final_layout; }
    auto found = std::ranges::find_if(
      images_, [declaration](tracked_image_state const &entry) -> bool { return entry.image == declaration.image; });
    if (found == images_.end()) {
      return first_image_use(declaration, step, family, queue, render_pass_transition, next, after);
    }
    auto &frontier = found->accesses;
    if ((frontier.anchor && frontier.anchor->step == step)
        || std::ranges::any_of(
          frontier.readers, [step](tracked_image const &reader) -> bool { return reader.step == step; })) {
      return fail(errc::invalid_argument, "multiple image uses in one pass require one combined declaration");
    }
    if (!same_range(found->range, declaration.range)) {
      return fail(errc::unsupported, "graph image uses require matching subresource ranges");
    }
    bool const transfer = found->family != family;
    if (render_pass_transition) { next.layout = found->layout; }
    bool const compatible_read = declaration.access == resource_access::read && !transfer
                                 && found->layout == next.layout && after.layout == next.layout
                                 && !render_pass_transition;
    if (compatible_read) {
      append_compatible_image_reader(*found, declaration, step, family, queue, next, after);
      return {};
    }
    return advance_image_state(*found, declaration, step, family, queue, next, after);
  }

public:
  explicit resource_state_tracker(std::size_t step_count) { plan_.steps.resize(step_count); }

  [[nodiscard]] auto use(image_use declaration, std::size_t step, std::uint32_t family, VkQueue queue = VK_NULL_HANDLE)
    -> status
  { return use_image(declaration, step, family, queue, false); }

  //! Synchronizes an attachment to the layout required at vkCmdBeginRenderPass.
  //! UNDEFINED leaves the layout transition to the render pass itself.
  [[nodiscard]] auto use_attachment(image_use declaration,
    VkImageLayout render_pass_initial_layout,
    std::size_t step,
    std::uint32_t family,
    VkQueue queue = VK_NULL_HANDLE) -> status
  {
    if (render_pass_initial_layout != VK_IMAGE_LAYOUT_UNDEFINED) { declaration.layout = render_pass_initial_layout; }
    return use_image(declaration, step, family, queue, render_pass_initial_layout == VK_IMAGE_LAYOUT_UNDEFINED);
  }

  [[nodiscard]] auto use(buffer_use declaration, std::size_t step, std::uint32_t family, VkQueue queue = VK_NULL_HANDLE)
    -> status
  {
    if (declaration.buffer == VK_NULL_HANDLE || declaration.size == 0) {
      return fail(errc::invalid_argument, "invalid graph buffer use");
    }
    if (!valid(declaration)) { return fail(errc::invalid_argument, "buffer usage and access are incompatible"); }
    usage_scope const next = buffer_scope(declaration);
    auto found = std::ranges::find_if(buffers_,
      [declaration](tracked_buffer_state const &entry) -> bool { return entry.buffer == declaration.buffer; });
    if (found == buffers_.end()) {
      auto const first_access =
        tracked_buffer{ .use = declaration, .scope = next, .step = step, .family = family, .queue = queue };
      bool const imported_read = declaration.access == resource_access::read;
      buffers_.push_back(tracked_buffer_state{ .buffer = declaration.buffer,
        .offset = declaration.offset,
        .size = declaration.size,
        .family = family,
        .accesses = { .anchor = imported_read ? std::optional<tracked_buffer>{} : std::optional{ first_access },
          .readers = imported_read ? std::vector{ first_access } : std::vector<tracked_buffer>{} } });
      return {};
    }
    auto &frontier = found->accesses;
    if ((frontier.anchor && frontier.anchor->step == step)
        || std::ranges::any_of(
          frontier.readers, [step](tracked_buffer const &reader) -> bool { return reader.step == step; })) {
      return fail(errc::invalid_argument, "multiple buffer uses in one pass require one combined declaration");
    }
    if (found->offset != declaration.offset || found->size != declaration.size) {
      return fail(errc::unsupported, "graph buffer uses require matching ranges");
    }
    bool const transfer = found->family != family;
    if (declaration.access == resource_access::read && !transfer) {
      if (frontier.anchor) {
        auto const &anchor = *frontier.anchor;
        add_dependency(anchor.step, step);
        if (anchor.use.access != resource_access::read) {
          auto const source = source_scope(access_frontier<tracked_buffer>{ .anchor = anchor, .readers = {} }, queue);
          plan_.steps.at(step).buffers_before.push_back(buffer_barrier_params{ .buffer = declaration.buffer,
            .offset = declaration.offset,
            .size = declaration.size,
            .src_stage = source.stage,
            .dst_stage = next.stage,
            .src_access = source.access,
            .dst_access = next.access });
        }
      }
      frontier.readers.push_back(
        tracked_buffer{ .use = declaration, .scope = next, .step = step, .family = family, .queue = queue });
      return {};
    }
    depend_on_frontier(frontier, step);
    if (transfer) {
      auto const handoff = collapse_frontier_for_transfer(frontier);
      if (!handoff) { return fail(errc::invalid_argument, "buffer transfer requires a previous access"); }
      plan_.steps.at(handoff->step)
        .buffers_after.push_back(buffer_barrier_params{ .buffer = declaration.buffer,
          .offset = declaration.offset,
          .size = declaration.size,
          .src_stage = handoff->scope.stage,
          .src_access = handoff->scope.access,
          .src_queue_family = found->family,
          .dst_queue_family = family });
      plan_.steps.at(step).buffers_before.push_back(buffer_barrier_params{ .buffer = declaration.buffer,
        .offset = declaration.offset,
        .size = declaration.size,
        .dst_stage = next.stage,
        .dst_access = next.access,
        .src_queue_family = found->family,
        .dst_queue_family = family });
    } else {
      auto const source = source_scope(frontier, queue);
      plan_.steps.at(step).buffers_before.push_back(buffer_barrier_params{ .buffer = declaration.buffer,
        .offset = declaration.offset,
        .size = declaration.size,
        .src_stage = source.stage,
        .dst_stage = next.stage,
        .src_access = source.access,
        .dst_access = next.access });
    }
    found->family = family;
    frontier.anchor =
      tracked_buffer{ .use = declaration, .scope = next, .step = step, .family = family, .queue = queue };
    frontier.readers.clear();
    return {};
  }

  template<class Uses>
  [[nodiscard]] auto
    use(Uses const &declarations, std::size_t step, std::uint32_t family, VkQueue queue = VK_NULL_HANDLE) -> status
  {
    for (image_use const declaration : declarations.images) { VKEXEC_TRY(use(declaration, step, family, queue)); }
    for (buffer_use const declaration : declarations.buffers) { VKEXEC_TRY(use(declaration, step, family, queue)); }
    return {};
  }

  [[nodiscard]] auto finish() && -> resource_sync_plan { return std::move(plan_); }
};

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_HPP
