#ifndef VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_HPP
#define VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_HPP

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
#if defined(VKEXEC_ENABLE_ALLOCATION_FAULT_INJECTION)
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
#if defined(VKEXEC_ENABLE_ALLOCATION_FAULT_INJECTION)
#ifndef VKEXEC_HAS_EXCEPTIONS
#error "VKEXEC_ENABLE_ALLOCATION_FAULT_INJECTION requires <vkexec/config.hpp> (VKEXEC_HAS_EXCEPTIONS)"
#endif
#if VKEXEC_HAS_EXCEPTIONS
  static inline thread_local int remaining{};

  static auto arm(int count) noexcept -> void { remaining = count; }

  static auto disarm() noexcept -> void { remaining = 0; }

  //! Throws on the Nth armed checkpoint (`arm(1)` → first call, `arm(2)` → second, …).
  static auto maybe_throw() -> void
  {
    if (remaining > 0 && --remaining == 0) { throw std::bad_alloc{}; }
  }
#else
  static auto arm(int) noexcept -> void {}

  static auto disarm() noexcept -> void {}

  static auto maybe_throw() noexcept -> void {}
#endif
#else
  static auto arm(int) noexcept -> void {}

  static auto disarm() noexcept -> void {}

  static auto maybe_throw() noexcept -> void {}
#endif
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
    for (auto const &gap : uncovered) {
      for (auto const &piece : subtract(gap, *common)) { remaining.push_back(piece); }
    }
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

  //! Restores `batch_` and rolls back uncommitted work on scope exit, including exception unwinding.
  class batch_transaction_guard
  {
    resource_state_tracker *owner_{ nullptr };
    batch_transaction batch_{};
    bool committed_{ false };

  public:
    explicit batch_transaction_guard(resource_state_tracker &owner) noexcept : owner_(&owner)
    {
      batch_.buffer_count = owner.buffers_.size();
      batch_.image_count = owner.images_.size();
      batch_.dependency_count = owner.plan_.dependencies.size();
      batch_.image_use_count = owner.plan_.image_uses.size();
      owner.batch_ = &batch_;
    }

    batch_transaction_guard(batch_transaction_guard const &) = delete;
    auto operator=(batch_transaction_guard const &) -> batch_transaction_guard & = delete;

    ~batch_transaction_guard() noexcept
    {
      if (committed_ || owner_ == nullptr) { return; }
      owner_->batch_ = nullptr;
      owner_->rollback_batch(batch_);
    }

    auto commit() noexcept -> void
    {
      owner_->batch_ = nullptr;
      committed_ = true;
    }
  };

  resource_sync_plan plan_;
  std::vector<tracked_image_resource> images_;
  std::vector<tracked_buffer_resource> buffers_;
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
      auto &point = plan_.steps[step.step];
      point.buffers_before.resize(step.buffers_before);
      point.buffers_after.resize(step.buffers_after);
      point.images_before.resize(step.images_before);
      point.images_after.resize(step.images_after);
    }
    for (auto iterator = batch.buffers.rbegin(); iterator != batch.buffers.rend(); ++iterator) {
      buffers_[iterator->index] = std::move(iterator->prior);
    }
    buffers_.resize(batch.buffer_count);
    for (auto iterator = batch.images.rbegin(); iterator != batch.images.rend(); ++iterator) {
      images_[iterator->index] = std::move(iterator->prior);
    }
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
    for (auto const &state : states) {
      if ((state.region.aspects & region.aspects) != 0 && state.region.aspects != region.aspects) {
        return fail(errc::unsupported, "cannot mix coupled and separate depth/stencil declarations for one image");
      }
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
        if (auto registered = register_extent(extent); !registered) { return fail(registered); }
      }
    }
    if constexpr (requires { declarations.buffer_extents; }) {
      for (auto const extent : declarations.buffer_extents) {
        if (auto registered = register_extent(extent); !registered) { return fail(registered); }
      }
    }
    for (image_use const declaration : declarations.images) {
      if (auto used = use(declaration, step, family, queue); !used) { return fail(used); }
    }
    for (buffer_use const declaration : declarations.buffers) {
      if (auto used = use(declaration, step, family, queue); !used) { return fail(used); }
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
