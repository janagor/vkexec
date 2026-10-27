#ifndef VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_PRIVATE_HPP
#define VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_PRIVATE_HPP

// Included inside resource_state_tracker after its state declarations.

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

  template<class Access>
  [[nodiscard]] static auto frontier_handoff_queue(access_frontier<Access> const &frontier) -> VkQueue
  {
    if (!frontier.readers.empty()) { return frontier.readers.back().queue; }
    if (frontier.anchor) { return frontier.anchor->queue; }
    return VK_NULL_HANDLE;
  }

  [[nodiscard]] static auto clamp_image_region(image_region &region, image_extent const &extent) -> status
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

  [[nodiscard]] static auto clamp_buffer_range(buffer_range &range, VkDeviceSize size) -> status
  {
    if (range.begin >= size || (range.end && *range.end > size)) {
      return fail(errc::invalid_argument, "graph buffer range exceeds registered extent");
    }
    if (!range.end) { range.end = size; }
    return {};
  }

  [[nodiscard]] static auto reject_mixed_aspect_declarations(std::vector<image_region_state> const &states,
    image_region const &region) -> status
  {
    if (std::ranges::any_of(states, [&region](image_region_state const &state) -> bool {
          return (state.region.aspects & region.aspects) != 0 && state.region.aspects != region.aspects;
        })) {
      return fail(errc::unsupported, "cannot mix coupled and separate depth/stencil declarations for one image");
    }
    return {};
  }

  [[nodiscard]] static auto validate_image_use_conflicts(std::vector<image_region_state> const &regions,
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

  [[nodiscard]] static auto validate_buffer_use_conflicts(std::vector<buffer_region_state> const &regions,
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

#endif// VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_PRIVATE_HPP
