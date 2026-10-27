#ifndef VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_HPP
#define VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_HPP

#include <vkexec/detail/resource_state_tracker_validation.hpp>

namespace vkexec::detail {

class resource_state_tracker
{
  //! Journals pre-existing mutated resources and barrier-vector append baselines.
  //! Newly appended resources are discarded by restoring `buffer_count` / `image_count`.
  struct batch_transaction
  {
    std::size_t buffer_count{};
    std::size_t image_count{};
    std::size_t dependency_count{};
    std::size_t image_use_count{};
    struct buffer_checkpoint
    {
      std::size_t index{};
      tracked_buffer_resource prior{};
    };
    struct image_checkpoint
    {
      std::size_t index{};
      tracked_image_resource prior{};
    };
    struct step_checkpoint
    {
      std::size_t step{};
      std::size_t buffers_before{};
      std::size_t buffers_after{};
      std::size_t images_before{};
      std::size_t images_after{};
    };
    std::vector<buffer_checkpoint> buffers;
    std::vector<image_checkpoint> images;
    std::vector<step_checkpoint> steps;
  };

  //! Restores the active journal and rolls back uncommitted work on scope exit, including exception unwinding.
  class batch_transaction_guard
  {
    resource_state_tracker *owner_{ nullptr };
    bool committed_{ false };

  public:
    explicit batch_transaction_guard(resource_state_tracker &owner) noexcept : owner_(&owner)
    {
      owner.batch_storage_.buffers.clear();
      owner.batch_storage_.images.clear();
      owner.batch_storage_.steps.clear();
      owner.batch_storage_.buffer_count = owner.buffers_.size();
      owner.batch_storage_.image_count = owner.images_.size();
      owner.batch_storage_.dependency_count = owner.plan_.dependencies.size();
      owner.batch_storage_.image_use_count = owner.plan_.image_uses.size();
      owner.batch_ = &owner.batch_storage_;
    }

    batch_transaction_guard(batch_transaction_guard const &) = delete;
    auto operator=(batch_transaction_guard const &) -> batch_transaction_guard & = delete;
    batch_transaction_guard(batch_transaction_guard &&) = delete;
    auto operator=(batch_transaction_guard &&) -> batch_transaction_guard & = delete;

    ~batch_transaction_guard() noexcept
    {
      if (!committed_) { rollback(); }
    }

    auto rollback() noexcept -> void
    {
      owner_->batch_ = nullptr;
      owner_->rollback_batch(owner_->batch_storage_);
      owner_->batch_storage_.buffers.clear();
      owner_->batch_storage_.images.clear();
      owner_->batch_storage_.steps.clear();
      committed_ = true;
    }

    auto commit() noexcept -> void
    {
      owner_->batch_ = nullptr;
      owner_->batch_storage_.buffers.clear();
      owner_->batch_storage_.images.clear();
      owner_->batch_storage_.steps.clear();
      committed_ = true;
    }
  };

  resource_sync_plan plan_;
  std::vector<tracked_image_resource> images_;
  std::vector<tracked_buffer_resource> buffers_;
  batch_transaction batch_storage_{};
  batch_transaction *batch_{ nullptr };

  auto remember_buffer(std::size_t index) -> void
  {
    if (batch_ == nullptr || index >= batch_->buffer_count) { return; }
    if (std::ranges::find_if(batch_->buffers,
          [index](batch_transaction::buffer_checkpoint const &entry) -> bool { return entry.index == index; })
        != batch_->buffers.end()) {
      return;
    }
    batch_->buffers.push_back(batch_transaction::buffer_checkpoint{ .index = index, .prior = buffers_.at(index) });
  }

  auto remember_image(std::size_t index) -> void
  {
    if (batch_ == nullptr || index >= batch_->image_count) { return; }
    if (std::ranges::find_if(batch_->images,
          [index](batch_transaction::image_checkpoint const &entry) -> bool { return entry.index == index; })
        != batch_->images.end()) {
      return;
    }
    batch_->images.push_back(batch_transaction::image_checkpoint{ .index = index, .prior = images_.at(index) });
  }

  auto remember_step(std::size_t step) -> void
  {
    if (batch_ == nullptr) { return; }
    if (std::ranges::find_if(
          batch_->steps, [step](batch_transaction::step_checkpoint const &entry) -> bool { return entry.step == step; })
        != batch_->steps.end()) {
      return;
    }
    auto const &point = plan_.steps.at(step);
    batch_->steps.push_back(batch_transaction::step_checkpoint{ .step = step,
      .buffers_before = point.buffers_before.size(),
      .buffers_after = point.buffers_after.size(),
      .images_before = point.images_before.size(),
      .images_after = point.images_after.size() });
  }

  //! Restores plan vectors by shrinking and replaces checkpointed resources; operations must not throw.
  auto rollback_batch(batch_transaction &batch) noexcept -> void
  {
    plan_.dependencies.resize(batch.dependency_count);
    plan_.image_uses.resize(batch.image_use_count);
    for (auto const &step : batch.steps) {
      auto &point = plan_.steps.at(step.step);
      point.buffers_before.resize(step.buffers_before);
      point.buffers_after.resize(step.buffers_after);
      point.images_before.resize(step.images_before);
      point.images_after.resize(step.images_after);
    }
    for (auto &buffer : std::views::reverse(batch.buffers)) { buffers_.at(buffer.index) = std::move(buffer.prior); }
    buffers_.resize(batch.buffer_count);
    for (auto &image : std::views::reverse(batch.images)) { images_.at(image.index) = std::move(image.prior); }
    images_.resize(batch.image_count);
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

  auto apply_image_region_uses(std::vector<image_region_state> &regions,
    image_region const &region,
    image_use const &declaration,
    std::size_t step,
    std::uint32_t family,
    VkQueue queue,
    bool exclusive,
    usage_scope next,
    usage_scope after,
    bool render_pass_transition) -> void
  {
    for (auto &state : regions) {
      if (!overlaps(state.region, region)) { continue; }
      auto scoped = declaration;
      scoped.range = subresource_range(state.region);
      if (!state.accesses.anchor && state.accesses.readers.empty()) {
        first_image_use(state, scoped, step, family, queue, render_pass_transition, next, after);
        continue;
      }
      auto before = next;
      if (render_pass_transition) { before.layout = state.layout; }
      bool const compatible_read = declaration.access == resource_access::read && (!exclusive || state.family == family)
                                   && state.layout == before.layout && after.layout == before.layout
                                   && !render_pass_transition;
      if (compatible_read) {
        append_compatible_image_reader(state, scoped, step, family, queue, before, after);
      } else {
        advance_image_state(state, scoped, step, family, queue, before, after, exclusive && state.family != family);
      }
    }
  }

  auto first_image_use(image_region_state &state,
    image_use declaration,
    std::size_t step,
    std::uint32_t family,
    VkQueue queue,
    bool render_pass_transition,
    usage_scope next,
    usage_scope after) -> void
  {
    if (render_pass_transition) { next.layout = declaration.initial_layout; }
    if (next.layout != VK_IMAGE_LAYOUT_UNDEFINED) {
      remember_step(step);
      resource_allocation_fault::maybe_throw();
      plan_.steps.at(step).images_before.push_back(image_barrier_params{ .image = declaration.image,
        .range = declaration.range,
        .old_layout = declaration.initial_layout,
        .new_layout = next.layout,
        .dst_stage = next.stage,
        .dst_access = next.access });
    }
    auto const first_access =
      tracked_image{ .access = declaration.access, .scope = after, .step = step, .family = family, .queue = queue };
    bool const imported_read = declaration.access == resource_access::read && declaration.initial_layout == next.layout
                               && after.layout == next.layout && !render_pass_transition;
    access_frontier<tracked_image> frontier{};
    if (imported_read) {
      frontier.readers.push_back(first_access);
    } else {
      frontier.anchor = first_access;
    }
    state.layout = after.layout;
    state.family = family;
    state.accesses = std::move(frontier);
  }

  auto append_compatible_image_reader(image_region_state &state,
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
      if (anchor.access != resource_access::read) {
        auto const source = source_scope(access_frontier<tracked_image>{ .anchor = anchor, .readers = {} }, queue);
        remember_step(step);
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
    auto const reader =
      tracked_image{ .access = declaration.access, .scope = after, .step = step, .family = family, .queue = queue };
    if (std::ranges::find(frontier.readers, reader) == frontier.readers.end()) { frontier.readers.push_back(reader); }
  }

  auto advance_image_state(image_region_state &state,
    image_use declaration,
    std::size_t step,
    std::uint32_t family,
    VkQueue queue,
    usage_scope next,
    usage_scope after,
    bool transfer) -> void
  {
    auto &frontier = state.accesses;
    depend_on_frontier(frontier, step);
    if (transfer) {
      auto *const handoff_queue = frontier_handoff_queue(frontier);
      auto const source = source_scope(frontier, handoff_queue);
      if (auto const handoff = collapse_frontier_for_transfer(frontier)) {
        remember_step(handoff->step);
        plan_.steps.at(handoff->step)
          .images_after.push_back(image_barrier_params{ .image = declaration.image,
            .range = declaration.range,
            .old_layout = state.layout,
            .new_layout = next.layout,
            .src_stage = source.stage,
            .src_access = source.access,
            .src_queue_family = state.family,
            .dst_queue_family = family });
      }
      remember_step(step);
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
      remember_step(step);
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
      tracked_image{ .access = declaration.access, .scope = after, .step = step, .family = family, .queue = queue };
    frontier.readers.clear();
    if (declaration.access == resource_access::read) { frontier.readers.push_back(*frontier.anchor); }
  }

  [[nodiscard]] auto
    use_image(image_use declaration, std::size_t step, std::uint32_t family, VkQueue queue, bool render_pass_transition)
      -> status
  {
    if (declaration.image == VK_NULL_HANDLE || step >= plan_.steps.size()) {
      return fail(errc::invalid_argument, "invalid graph image use");
    }
    if (!valid(declaration)) { return fail(errc::invalid_argument, "image usage and access are incompatible"); }
    VKEXEC_TRY_ASSIGN(region, normalize(declaration.range));
    usage_scope const next = image_scope(declaration);
    usage_scope after = next;
    if (declaration.final_layout != VK_IMAGE_LAYOUT_UNDEFINED) { after.layout = declaration.final_layout; }
    auto found = std::ranges::find_if(
      images_, [declaration](tracked_image_resource const &entry) -> bool { return entry.image == declaration.image; });
    bool exclusive = true;
    if (found != images_.end()) {
      if (auto const extent = found->extent) {
        exclusive = extent->sharing_mode == VK_SHARING_MODE_EXCLUSIVE;
        VKEXEC_TRY(clamp_image_region(region, *extent));
      }
      // A combined depth/stencil declaration is an indivisible layout domain.
      // Mixing it with separate-aspect declarations needs device/format metadata.
      VKEXEC_TRY(reject_mixed_aspect_declarations(found->regions, region));
    }
    std::vector<image_region_state> regions;
    if (found != images_.end()) {
      VKEXEC_TRY_ASSIGN(partitioned, partition_image_regions(found->regions, region));
      regions = std::move(partitioned);
    } else {
      regions.push_back(image_region_state{ .region = region, .accesses = {} });
    }
    // Validate every piece, including uncovered subresources, before modifying the plan.
    VKEXEC_TRY(validate_image_use_conflicts(
      regions, region, declaration, step, family, queue, exclusive, next, after, render_pass_transition));
    if (found != images_.end()) { remember_image(static_cast<std::size_t>(std::distance(images_.begin(), found))); }
    apply_image_region_uses(
      regions, region, declaration, step, family, queue, exclusive, next, after, render_pass_transition);
    merge_image_regions(regions);
    if (found == images_.end()) {
      resource_allocation_fault::maybe_throw();
      images_.push_back(
        tracked_image_resource{ .image = declaration.image, .regions = std::move(regions), .extent = std::nullopt });
    } else {
      found->regions = std::move(regions);
    }
    plan_.image_uses.push_back(resource_sync_plan::image_access{ .image = declaration.image,
      .usage = declaration.usage,
      .final_layout = declaration.final_layout,
      .step = step,
      .range = declaration.range });
    return {};
  }

  auto advance_buffer_region(buffer_region_state &state,
    buffer_use declaration,
    std::size_t step,
    std::uint32_t family,
    VkQueue queue,
    bool transfer) -> void
  {
    declaration.offset = state.range.begin;
    declaration.size = region_size(state.range);
    auto const next = buffer_scope(declaration);
    auto &frontier = state.accesses;
    if (!frontier.anchor && frontier.readers.empty()) {
      auto const first_access =
        tracked_buffer{ .access = declaration.access, .scope = next, .step = step, .family = family, .queue = queue };
      state.family = family;
      if (declaration.access == resource_access::read) {
        frontier.readers.push_back(first_access);
      } else {
        frontier.anchor = first_access;
      }
      return;
    }
    if (declaration.access == resource_access::read && !transfer) {
      if (frontier.anchor) {
        auto const &anchor = *frontier.anchor;
        add_dependency(anchor.step, step);
        if (anchor.access != resource_access::read) {
          auto const source = source_scope(access_frontier<tracked_buffer>{ .anchor = anchor, .readers = {} }, queue);
          remember_step(step);
          plan_.steps.at(step).buffers_before.push_back(buffer_barrier_params{ .buffer = declaration.buffer,
            .offset = declaration.offset,
            .size = declaration.size,
            .src_stage = source.stage,
            .dst_stage = next.stage,
            .src_access = source.access,
            .dst_access = next.access });
        }
      }
      auto const reader =
        tracked_buffer{ .access = declaration.access, .scope = next, .step = step, .family = family, .queue = queue };
      if (std::ranges::find(frontier.readers, reader) == frontier.readers.end()) { frontier.readers.push_back(reader); }
      return;
    }
    depend_on_frontier(frontier, step);
    if (transfer) {
      auto *const handoff_queue = frontier_handoff_queue(frontier);
      auto const source = source_scope(frontier, handoff_queue);
      if (auto const handoff = collapse_frontier_for_transfer(frontier)) {
        remember_step(handoff->step);
        plan_.steps.at(handoff->step)
          .buffers_after.push_back(buffer_barrier_params{ .buffer = declaration.buffer,
            .offset = declaration.offset,
            .size = declaration.size,
            .src_stage = source.stage,
            .src_access = source.access,
            .src_queue_family = state.family,
            .dst_queue_family = family });
      }
      remember_step(step);
      plan_.steps.at(step).buffers_before.push_back(buffer_barrier_params{ .buffer = declaration.buffer,
        .offset = declaration.offset,
        .size = declaration.size,
        .dst_stage = next.stage,
        .dst_access = next.access,
        .src_queue_family = state.family,
        .dst_queue_family = family });
    } else {
      auto const source = source_scope(frontier, queue);
      remember_step(step);
      plan_.steps.at(step).buffers_before.push_back(buffer_barrier_params{ .buffer = declaration.buffer,
        .offset = declaration.offset,
        .size = declaration.size,
        .src_stage = source.stage,
        .dst_stage = next.stage,
        .src_access = source.access,
        .dst_access = next.access });
    }
    state.family = family;
    frontier.anchor =
      tracked_buffer{ .access = declaration.access, .scope = next, .step = step, .family = family, .queue = queue };
    frontier.readers.clear();
  }

public:
  explicit resource_state_tracker(std::size_t step_count) { plan_.steps.resize(step_count); }

  [[nodiscard]] auto register_extent(buffer_extent extent) -> status
  {
    if (extent.buffer == VK_NULL_HANDLE || extent.size == 0
        || (extent.sharing_mode != VK_SHARING_MODE_EXCLUSIVE && extent.sharing_mode != VK_SHARING_MODE_CONCURRENT)) {
      return fail(errc::invalid_argument, "invalid graph buffer extent");
    }
    auto found = std::ranges::find_if(
      buffers_, [extent](tracked_buffer_resource const &resource) -> bool { return resource.buffer == extent.buffer; });
    if (found == buffers_.end()) {
      resource_allocation_fault::maybe_throw();
      buffers_.push_back(tracked_buffer_resource{
        .buffer = extent.buffer, .regions = {}, .size = extent.size, .sharing_mode = extent.sharing_mode });
      return {};
    }
    if (found->size == extent.size && found->sharing_mode == extent.sharing_mode) { return {}; }
    return fail(errc::invalid_argument, "buffer extent must be registered consistently before its first use");
  }

  [[nodiscard]] auto register_extent(image_extent extent) -> status
  {
    if (extent.image == VK_NULL_HANDLE || extent.mip_levels == 0 || extent.array_layers == 0
        || extent.mip_levels == VK_REMAINING_MIP_LEVELS || extent.array_layers == VK_REMAINING_ARRAY_LAYERS
        || (extent.sharing_mode != VK_SHARING_MODE_EXCLUSIVE && extent.sharing_mode != VK_SHARING_MODE_CONCURRENT)) {
      return fail(errc::invalid_argument, "invalid graph image extent");
    }
    auto found = std::ranges::find_if(
      images_, [extent](tracked_image_resource const &resource) -> bool { return resource.image == extent.image; });
    if (found == images_.end()) {
      resource_allocation_fault::maybe_throw();
      images_.push_back(tracked_image_resource{ .image = extent.image, .regions = {}, .extent = extent });
      return {};
    }
    if (found->extent && found->extent->mip_levels == extent.mip_levels
        && found->extent->array_layers == extent.array_layers && found->extent->sharing_mode == extent.sharing_mode) {
      return {};
    }
    return fail(errc::invalid_argument, "image extent must be registered consistently before its first use");
  }

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
    if (declaration.buffer == VK_NULL_HANDLE || step >= plan_.steps.size()) {
      return fail(errc::invalid_argument, "invalid graph buffer use");
    }
    if (!valid(declaration)) { return fail(errc::invalid_argument, "buffer usage and access are incompatible"); }
    VKEXEC_TRY_ASSIGN(range, normalize(declaration));
    auto found = std::ranges::find_if(buffers_,
      [declaration](tracked_buffer_resource const &entry) -> bool { return entry.buffer == declaration.buffer; });
    bool created = false;
    if (found != buffers_.end()) {
      if (auto const size = found->size) { VKEXEC_TRY(clamp_buffer_range(range, *size)); }
    } else {
      resource_allocation_fault::maybe_throw();
      buffers_.push_back(tracked_buffer_resource{ .buffer = declaration.buffer, .regions = {}, .size = std::nullopt });
      found = std::prev(buffers_.end());
      created = true;
    }
    // Validate every intersection before splitting or adding barriers and dependencies.
    VKEXEC_TRY(validate_buffer_use_conflicts(found->regions, range, declaration, step, family, queue));
    VKEXEC_TRY_ASSIGN(partitioned, partition_buffer_regions(found->regions, range));
    if (!created) { remember_buffer(static_cast<std::size_t>(std::distance(buffers_.begin(), found))); }
    auto regions = std::move(partitioned);
    for (auto &region : regions) {
      if (overlaps(region.range, range)) {
        advance_buffer_region(region,
          declaration,
          step,
          family,
          queue,
          found->sharing_mode == VK_SHARING_MODE_EXCLUSIVE && region.family != family);
      }
    }
    merge_buffer_regions(regions);
    found->regions = std::move(regions);
    return {};
  }

  //! Applies every declaration atomically: on any failure the tracker is left unchanged.
  template<class Uses>
  [[nodiscard]] auto
    use(Uses const &declarations, std::size_t step, std::uint32_t family, VkQueue queue = VK_NULL_HANDLE) -> status
  {
    batch_transaction_guard guard{ *this };
    if constexpr (requires { declarations.image_extents; }) {
      for (auto const extent : declarations.image_extents) {
        if (auto registered = register_extent(extent); !registered) {
          guard.rollback();
          return fail(registered);
        }
      }
    }
    if constexpr (requires { declarations.buffer_extents; }) {
      for (auto const extent : declarations.buffer_extents) {
        if (auto registered = register_extent(extent); !registered) {
          guard.rollback();
          return fail(registered);
        }
      }
    }
    for (image_use const declaration : declarations.images) {
      if (auto used = use(declaration, step, family, queue); !used) {
        guard.rollback();
        return fail(used);
      }
    }
    for (buffer_use const declaration : declarations.buffers) {
      if (auto used = use(declaration, step, family, queue); !used) {
        guard.rollback();
        return fail(used);
      }
    }
    guard.commit();
    return {};
  }

  [[nodiscard]] auto region_counts() const noexcept -> resource_region_counts
  {
    resource_region_counts counts{};
    for (auto const &resource : buffers_) { counts.buffers += resource.regions.size(); }
    for (auto const &resource : images_) { counts.images += resource.regions.size(); }
    return counts;
  }

  [[nodiscard]] auto finish() && -> resource_sync_plan
  {
    for (auto &step : plan_.steps) {
      coalesce_barriers(step.buffers_before);
      coalesce_barriers(step.buffers_after);
      coalesce_barriers(step.images_before);
      coalesce_barriers(step.images_after);
    }
    return std::move(plan_);
  }
};

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_HPP
