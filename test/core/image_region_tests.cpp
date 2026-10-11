// NOLINTBEGIN(readability-function-cognitive-complexity)

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vkexec/config.hpp>
#include <vkexec/detail/resource_ranges.hpp>
#include <vkexec/detail/resource_state_tracker.hpp>
#include <vkexec/resource_use.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#if defined(VKEXEC_ENABLE_ALLOCATION_FAULT_INJECTION)
#ifndef VKEXEC_HAS_EXCEPTIONS
#error "VKEXEC_ENABLE_ALLOCATION_FAULT_INJECTION requires <vkexec/config.hpp> (VKEXEC_HAS_EXCEPTIONS)"
#endif
#if VKEXEC_HAS_EXCEPTIONS
#include <new>
#endif
#endif
#include <optional>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace {

constexpr std::uint32_t k_base = 0;
constexpr std::uint32_t k_next = 1;
constexpr std::uint32_t k_middle = 2;
constexpr std::uint32_t k_last = 3;
constexpr std::uint32_t k_extent = 4;
constexpr std::uint32_t k_one = 1;
constexpr std::size_t k_first_step = 0;
constexpr std::size_t k_second_step = 1;
constexpr std::size_t k_third_step = 2;
constexpr std::size_t k_fourth_step = 3;
constexpr std::size_t k_step_count = 4;
constexpr std::size_t k_one_barrier = 1;
constexpr std::size_t k_two_barriers = 2;
constexpr std::uint32_t k_family = 0;
constexpr std::uint32_t k_other_family = 1;
constexpr VkImageAspectFlags k_depth_stencil = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;

[[nodiscard]] auto fake_image(void *storage) -> VkImage { return static_cast<VkImage>(storage); }

[[nodiscard]] constexpr auto range(std::uint32_t mip = k_base,
  std::uint32_t mips = k_one,
  std::uint32_t layer = k_base,
  std::uint32_t layers = k_one,
  VkImageAspectFlags aspects = VK_IMAGE_ASPECT_COLOR_BIT) -> VkImageSubresourceRange
{
  return {
    .aspectMask = aspects, .baseMipLevel = mip, .levelCount = mips, .baseArrayLayer = layer, .layerCount = layers
  };
}

[[nodiscard]] auto has_edge(vkexec::detail::resource_sync_plan const &plan, std::size_t source, std::size_t destination)
  -> bool
{ return std::ranges::find(plan.dependencies, std::pair{ source, destination }) != plan.dependencies.end(); }

auto require_range(VkImageSubresourceRange actual, VkImageSubresourceRange expected) -> void
{
  REQUIRE(actual.aspectMask == expected.aspectMask);
  REQUIRE(actual.baseMipLevel == expected.baseMipLevel);
  REQUIRE(actual.levelCount == expected.levelCount);
  REQUIRE(actual.baseArrayLayer == expected.baseArrayLayer);
  REQUIRE(actual.layerCount == expected.layerCount);
}

}// namespace

TEST_CASE("image rectangle subtraction preserves four surrounding regions", "[vkexec][ranges]")
{
  using vkexec::detail::image_region_state;
  using vkexec::detail::merge_image_regions;
  using vkexec::detail::normalize;
  using vkexec::detail::overlaps;
  using vkexec::detail::partition_image_regions;
  using vkexec::detail::subtract;

  auto const whole = normalize(range(k_base, k_extent, k_base, k_extent));
  auto const middle = normalize(range(k_next, k_middle, k_next, k_middle));
  REQUIRE(whole.has_value());
  REQUIRE(middle.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access) -- Catch2's REQUIRE above checks engagement.
  auto const pieces = subtract(*whole, *middle);
  REQUIRE(pieces.size() == k_extent);
  for (std::size_t index{}; index < pieces.size(); ++index) {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access) -- Catch2's REQUIRE above checks engagement.
    REQUIRE_FALSE(overlaps(pieces.at(index), *middle));
    for (std::size_t other = index + 1; other < pieces.size(); ++other) {
      REQUIRE_FALSE(overlaps(pieces.at(index), pieces.at(other)));
    }
  }
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access) -- Catch2's REQUIRE above checks engagement.
  std::vector<image_region_state> states{ image_region_state{ .region = *whole, .accesses = {} } };
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access) -- Catch2's REQUIRE above checks engagement.
  auto partitioned = partition_image_regions(states, *middle);
  REQUIRE(partitioned.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access) -- Catch2's REQUIRE above checks engagement.
  states = std::move(*partitioned);
  merge_image_regions(states);
  REQUIRE(states.size() == k_one_barrier);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access) -- Catch2's REQUIRE above checks engagement.
  REQUIRE(states.front().region == *whole);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access) -- Catch2's REQUIRE above checks engagement.
  REQUIRE(subtract(*whole, *whole).empty());
}

TEST_CASE("image partitioning accepts disjoint finite mips that share remaining layers", "[vkexec][ranges]")
{
  using vkexec::detail::image_region_state;
  using vkexec::detail::normalize;
  using vkexec::detail::partition_image_regions;

  auto const mip0 = normalize(range(k_base, k_one, k_base, VK_REMAINING_ARRAY_LAYERS));
  auto const mip1 = normalize(range(k_next, k_one, k_base, VK_REMAINING_ARRAY_LAYERS));
  REQUIRE(mip0.has_value());
  REQUIRE(mip1.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access) -- Catch2's REQUIRE above checks engagement.
  std::vector<image_region_state> states{ image_region_state{ .region = *mip0, .accesses = {} } };
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access) -- Catch2's REQUIRE above checks engagement.
  auto const partitioned = partition_image_regions(states, *mip1);
  REQUIRE(partitioned.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access) -- Catch2's REQUIRE above checks engagement.
  REQUIRE(partitioned->size() == k_two_barriers);
}

TEST_CASE("image range normalization preserves symbolic remaining counts", "[vkexec][ranges]")
{
  using vkexec::detail::normalize;
  using vkexec::detail::subresource_range;

  auto const remaining = range(k_next, VK_REMAINING_MIP_LEVELS, k_middle, VK_REMAINING_ARRAY_LAYERS);
  auto const normalized = normalize(remaining);
  REQUIRE(normalized.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access) -- Catch2's REQUIRE above checks engagement.
  REQUIRE_FALSE(normalized->mips.end);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access) -- Catch2's REQUIRE above checks engagement.
  REQUIRE_FALSE(normalized->layers.end);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access) -- Catch2's REQUIRE above checks engagement.
  require_range(subresource_range(*normalized), remaining);
  REQUIRE_FALSE(normalize(range(k_base, 0)));
  REQUIRE_FALSE(normalize(range(k_base, k_one, k_base, 0)));
  REQUIRE_FALSE(normalize(range(std::numeric_limits<std::uint32_t>::max() - k_next, k_extent)));
  REQUIRE_FALSE(normalize(range(k_base, k_one, std::numeric_limits<std::uint32_t>::max(), VK_REMAINING_ARRAY_LAYERS)));
  REQUIRE_FALSE(normalize(range(k_base, k_one, k_base, k_one, VK_IMAGE_ASPECT_PLANE_0_BIT)));
}

TEST_CASE("disjoint mips and layers stay independent within a pass", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(
    tracker.use(vkexec::write(image, range(), vkexec::image_usage::transfer_destination), k_first_step, k_family));
  REQUIRE(tracker.use(
    vkexec::read(image, range(k_next), vkexec::image_usage::sampled_compute, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
    k_first_step,
    k_family));
  REQUIRE(tracker.use(vkexec::write(image, range(k_base, k_one, k_next), vkexec::image_usage::storage_compute),
    k_second_step,
    k_other_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.dependencies.empty());
  REQUIRE(plan.steps.at(k_first_step).images_after.empty());
  REQUIRE(plan.steps.at(k_second_step).images_before.front().src_queue_family == VK_QUEUE_FAMILY_IGNORED);
  REQUIRE(plan.image_uses.size() == k_third_step + 1);
  require_range(plan.image_uses.at(k_second_step).range, range(k_next));
}

TEST_CASE("partial image layers synchronize only their intersection and initialize gaps", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(
    tracker.use(vkexec::write(image, range(k_base, k_one, k_base, k_middle), vkexec::image_usage::transfer_destination),
      k_first_step,
      k_family));
  REQUIRE(tracker.use(
    vkexec::read(
      image, range(k_base, k_one, k_next, k_middle), vkexec::image_usage::sampled_compute, VK_IMAGE_LAYOUT_GENERAL),
    k_second_step,
    k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(has_edge(plan, k_first_step, k_second_step));
  auto const &barriers = plan.steps.at(k_second_step).images_before;
  REQUIRE(barriers.size() == k_two_barriers);
  for (auto const &barrier : barriers) {
    REQUIRE(barrier.new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    REQUIRE(barrier.range.layerCount == k_one);
    if (barrier.range.baseArrayLayer == k_next) {
      REQUIRE(barrier.old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    } else {
      REQUIRE(barrier.range.baseArrayLayer == k_middle);
      REQUIRE(barrier.old_layout == VK_IMAGE_LAYOUT_GENERAL);
    }
  }
}

TEST_CASE("spanning image use retains different old layouts and predecessors", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(
    tracker.use(vkexec::write(image, range(), vkexec::image_usage::transfer_destination), k_first_step, k_family));
  REQUIRE(
    tracker.use(vkexec::write(image, range(k_next), vkexec::image_usage::storage_compute), k_second_step, k_family));
  REQUIRE(tracker.use(
    vkexec::read(image, range(k_base, k_middle), vkexec::image_usage::sampled_fragment), k_third_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(has_edge(plan, k_first_step, k_third_step));
  REQUIRE(has_edge(plan, k_second_step, k_third_step));
  REQUIRE_FALSE(has_edge(plan, k_first_step, k_second_step));
  auto const &barriers = plan.steps.at(k_third_step).images_before;
  REQUIRE(barriers.size() == k_two_barriers);
  for (auto const &barrier : barriers) {
    REQUIRE(barrier.range.levelCount == k_one);
    REQUIRE(barrier.old_layout
            == (barrier.range.baseMipLevel == k_base ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL));
  }
}

TEST_CASE("invalid image gap reads do not mutate tracked layouts or plan records", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(
    tracker.use(vkexec::write(image, range(), vkexec::image_usage::transfer_destination), k_first_step, k_family));
  REQUIRE_FALSE(tracker.use(
    vkexec::read(image, range(k_base, k_middle), vkexec::image_usage::sampled_fragment), k_second_step, k_family));
  REQUIRE(tracker.use(vkexec::read(image, range(), vkexec::image_usage::sampled_fragment), k_third_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_second_step).images_before.empty());
  REQUIRE(plan.image_uses.size() == k_two_barriers);
  REQUIRE_FALSE(has_edge(plan, k_first_step, k_second_step));
  REQUIRE(plan.steps.at(k_third_step).images_before.front().old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
}

TEST_CASE("partial image transfer uses matching releases and acquires", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(
    tracker.use(vkexec::write(image, range(k_base, k_extent, k_base, k_extent), vkexec::image_usage::storage_compute),
      k_first_step,
      k_family));
  auto const middle = range(k_next, k_middle, k_next, k_middle);
  REQUIRE(
    tracker.use(vkexec::read(image, middle, vkexec::image_usage::sampled_compute), k_second_step, k_other_family));
  REQUIRE(tracker.use(vkexec::read(image, range(), vkexec::image_usage::sampled_compute), k_third_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_first_step).images_after.size() == k_one_barrier);
  auto const &release = plan.steps.at(k_first_step).images_after.front();
  auto const &acquire = plan.steps.at(k_second_step).images_before.front();
  require_range(release.range, middle);
  require_range(acquire.range, middle);
  REQUIRE(release.old_layout == acquire.old_layout);
  REQUIRE(release.new_layout == acquire.new_layout);
  REQUIRE(release.src_queue_family == k_family);
  REQUIRE(release.dst_queue_family == k_other_family);
  REQUIRE(acquire.src_queue_family == release.src_queue_family);
  REQUIRE(acquire.dst_queue_family == release.dst_queue_family);
  REQUIRE(plan.steps.at(k_third_step).images_before.front().src_queue_family == VK_QUEUE_FAMILY_IGNORED);
  REQUIRE_FALSE(has_edge(plan, k_second_step, k_third_step));
}

TEST_CASE("render-pass final layouts affect only the declared image region", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(tracker.use(
    vkexec::write(image, range(k_base, k_middle), vkexec::image_usage::storage_compute), k_first_step, k_family));
  auto attachment = vkexec::write(image, range(k_next), vkexec::image_usage::color_attachment);
  attachment.final_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  REQUIRE(tracker.use_attachment(attachment, VK_IMAGE_LAYOUT_UNDEFINED, k_second_step, k_family));
  REQUIRE(tracker.use(
    vkexec::read(image, range(k_base, k_middle), vkexec::image_usage::sampled_fragment), k_third_step, k_family));
  auto const plan = std::move(tracker).finish();
  auto const &before = plan.steps.at(k_second_step).images_before.front();
  require_range(before.range, range(k_next));
  REQUIRE(before.old_layout == VK_IMAGE_LAYOUT_GENERAL);
  REQUIRE(before.new_layout == VK_IMAGE_LAYOUT_GENERAL);
  REQUIRE(plan.steps.at(k_third_step).images_before.size() == k_two_barriers);
  for (auto const &barrier : plan.steps.at(k_third_step).images_before) {
    REQUIRE(barrier.old_layout
            == (barrier.range.baseMipLevel == k_base ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL));
  }
}

TEST_CASE("unregistered remaining image ranges reject finite splits of open tails", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  auto const whole = range(k_base, VK_REMAINING_MIP_LEVELS, k_base, VK_REMAINING_ARRAY_LAYERS);
  REQUIRE(tracker.use(vkexec::write(image, whole, vkexec::image_usage::storage_compute), k_first_step, k_family));
  REQUIRE_FALSE(tracker.use(
    vkexec::write(image, range(k_next, k_middle, k_next, k_middle), vkexec::image_usage::transfer_destination),
    k_second_step,
    k_family));
  REQUIRE(tracker.region_counts().images == k_one_barrier);
}

TEST_CASE("unregistered finite image uses reject later remaining-count extensions", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(tracker.use(vkexec::write(image, range(k_base, k_one, k_base, k_one), vkexec::image_usage::storage_compute),
    k_first_step,
    k_family));
  auto const remaining = range(k_base, VK_REMAINING_MIP_LEVELS, k_base, VK_REMAINING_ARRAY_LAYERS);
  auto const failed =
    tracker.use(vkexec::write(image, remaining, vkexec::image_usage::transfer_destination), k_second_step, k_family);
  REQUIRE_FALSE(failed);
  REQUIRE(failed.error().code == vkexec::make_error_code(vkexec::errc::invalid_argument));
  REQUIRE(failed.error().message().find("open suffix") != std::string::npos);
  REQUIRE(tracker.region_counts().images == k_one_barrier);
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_second_step).images_before.empty());
  REQUIRE_FALSE(has_edge(plan, k_first_step, k_second_step));
}

TEST_CASE("disjoint finite image mips with remaining layers are accepted without registration",
  "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  auto const mip0 = range(k_base, k_one, k_base, VK_REMAINING_ARRAY_LAYERS);
  auto const mip1 = range(k_next, k_one, k_base, VK_REMAINING_ARRAY_LAYERS);
  REQUIRE(tracker.use(vkexec::write(image, mip0, vkexec::image_usage::storage_compute), k_first_step, k_family));
  REQUIRE(tracker.use(vkexec::write(image, mip1, vkexec::image_usage::storage_compute), k_second_step, k_family));
  REQUIRE(tracker.region_counts().images == k_two_barriers);
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.dependencies.empty());
  REQUIRE_FALSE(has_edge(plan, k_first_step, k_second_step));
  REQUIRE(plan.steps.at(k_second_step).images_before.size() == k_one_barrier);
  require_range(plan.steps.at(k_second_step).images_before.front().range, mip1);
  REQUIRE(plan.steps.at(k_second_step).images_before.front().old_layout == VK_IMAGE_LAYOUT_UNDEFINED);
}

TEST_CASE("independent depth and stencil remaining-layer ranges are accepted without registration",
  "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  auto const depth = range(k_base, k_one, k_base, VK_REMAINING_ARRAY_LAYERS, VK_IMAGE_ASPECT_DEPTH_BIT);
  auto const stencil = range(k_base, k_one, k_base, VK_REMAINING_ARRAY_LAYERS, VK_IMAGE_ASPECT_STENCIL_BIT);
  REQUIRE(tracker.use(vkexec::write(image, depth, vkexec::image_usage::depth_attachment), k_first_step, k_family));
  REQUIRE(tracker.use(vkexec::write(image, stencil, vkexec::image_usage::depth_attachment), k_second_step, k_family));
  REQUIRE(tracker.region_counts().images == k_two_barriers);
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.dependencies.empty());
}

TEST_CASE("registered finite image uses accept later remaining-count extensions", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(
    tracker.register_extent(vkexec::image_extent{ .image = image, .mip_levels = k_extent, .array_layers = k_extent }));
  REQUIRE(
    tracker.use(vkexec::write(image, range(k_base, k_extent, k_base, k_extent), vkexec::image_usage::storage_compute),
      k_first_step,
      k_family));
  auto const remaining = range(k_base, VK_REMAINING_MIP_LEVELS, k_base, VK_REMAINING_ARRAY_LAYERS);
  REQUIRE(tracker.use(vkexec::read(image, remaining, vkexec::image_usage::sampled_compute), k_second_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(has_edge(plan, k_first_step, k_second_step));
  REQUIRE(plan.steps.at(k_second_step).images_before.size() == k_one_barrier);
  require_range(plan.steps.at(k_second_step).images_before.front().range, range(k_base, k_extent, k_base, k_extent));
  REQUIRE(plan.steps.at(k_second_step).images_before.front().range.levelCount != VK_REMAINING_MIP_LEVELS);
  REQUIRE(plan.steps.at(k_second_step).images_before.front().range.layerCount != VK_REMAINING_ARRAY_LAYERS);
}

TEST_CASE("registered remaining image ranges clamp away empty open tails", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(
    tracker.register_extent(vkexec::image_extent{ .image = image, .mip_levels = k_extent, .array_layers = k_extent }));
  auto const whole = range(k_base, VK_REMAINING_MIP_LEVELS, k_base, VK_REMAINING_ARRAY_LAYERS);
  REQUIRE(tracker.use(vkexec::write(image, whole, vkexec::image_usage::storage_compute), k_first_step, k_family));
  REQUIRE(tracker.use(
    vkexec::write(image, range(k_next, k_middle, k_next, k_middle), vkexec::image_usage::transfer_destination),
    k_second_step,
    k_family));
  auto const tail = range(k_last, VK_REMAINING_MIP_LEVELS, k_last, VK_REMAINING_ARRAY_LAYERS);
  REQUIRE(tracker.use(vkexec::read(image, tail, vkexec::image_usage::sampled_compute), k_third_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(has_edge(plan, k_first_step, k_third_step));
  REQUIRE_FALSE(has_edge(plan, k_second_step, k_third_step));
  REQUIRE(plan.steps.at(k_third_step).images_before.size() == k_one_barrier);
  require_range(plan.steps.at(k_third_step).images_before.front().range, range(k_last, k_one, k_last, k_one));
}

TEST_CASE("depth stencil declarations preserve coupled barriers or separate aspects", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  SECTION("combined aspects stay coupled")
  {
    vkexec::detail::resource_state_tracker tracker{ k_step_count };
    auto const both = range(k_base, k_middle, k_base, k_one, k_depth_stencil);
    REQUIRE(tracker.use(vkexec::write(image, both, vkexec::image_usage::depth_attachment), k_first_step, k_family));
    auto const selected = range(k_next, k_one, k_base, k_one, k_depth_stencil);
    REQUIRE(tracker.use(vkexec::read(image, selected, vkexec::image_usage::sampled_fragment), k_second_step, k_family));
    REQUIRE_FALSE(tracker.use(
      vkexec::write(
        image, range(k_base, k_one, k_base, k_one, VK_IMAGE_ASPECT_DEPTH_BIT), vkexec::image_usage::depth_attachment),
      k_third_step,
      k_family));
    auto const plan = std::move(tracker).finish();
    require_range(plan.steps.at(k_second_step).images_before.front().range, selected);
  }
  SECTION("separate aspects require caller-enabled separate depth stencil layouts")
  {
    vkexec::detail::resource_state_tracker tracker{ k_step_count };
    REQUIRE(tracker.use(
      vkexec::write(
        image, range(k_base, k_one, k_base, k_one, VK_IMAGE_ASPECT_DEPTH_BIT), vkexec::image_usage::depth_attachment),
      k_first_step,
      k_family));
    REQUIRE(tracker.use(
      vkexec::write(
        image, range(k_base, k_one, k_base, k_one, VK_IMAGE_ASPECT_STENCIL_BIT), vkexec::image_usage::depth_attachment),
      k_first_step,
      k_family));
    auto const plan = std::move(tracker).finish();
    REQUIRE(plan.dependencies.empty());
    REQUIRE(plan.steps.at(k_first_step).images_before.size() == k_two_barriers);
  }
}

TEST_CASE("overlapping conflicting image declarations in one pass are rejected atomically", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(
    tracker.use(vkexec::write(image, range(k_next), vkexec::image_usage::storage_compute), k_first_step, k_family));
  REQUIRE_FALSE(tracker.use(
    vkexec::read(image, range(k_base, k_middle), vkexec::image_usage::sampled_compute, VK_IMAGE_LAYOUT_GENERAL),
    k_first_step,
    k_family));
  REQUIRE(tracker.use(vkexec::read(image, range(k_base), vkexec::image_usage::sampled_compute, VK_IMAGE_LAYOUT_GENERAL),
    k_second_step,
    k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.dependencies.empty());
  REQUIRE(plan.steps.at(k_first_step).images_before.size() == k_one_barrier);
}

TEST_CASE("image rectangles match a deterministic subresource cell reference", "[vkexec][pass][ranges]")
{
  constexpr std::size_t k_operation_count = 160;
  constexpr std::uint32_t k_seed = 0x1A6E5U;
  constexpr std::array k_aspects{ VkImageAspectFlags{ VK_IMAGE_ASPECT_DEPTH_BIT },
    VkImageAspectFlags{ VK_IMAGE_ASPECT_STENCIL_BIT } };
  constexpr std::size_t k_cell_count = k_aspects.size() * k_extent * k_extent;
  constexpr std::array k_accesses{
    vkexec::resource_access::read, vkexec::resource_access::write, vkexec::resource_access::read_write
  };
  struct cell_state
  {
    bool initialized{};
    VkImageLayout layout{ VK_IMAGE_LAYOUT_GENERAL };
    std::optional<std::size_t> anchor;
    std::vector<std::size_t> readers;
  };
  struct cell_barrier
  {
    bool present{};
    VkImageLayout old_layout{ VK_IMAGE_LAYOUT_UNDEFINED };
    VkImageLayout new_layout{ VK_IMAGE_LAYOUT_UNDEFINED };
    VkPipelineStageFlags2 source_stage{};
    VkAccessFlags2 source_access{};
  };
  auto const cell_index = [](std::size_t aspect, std::uint32_t mip, std::uint32_t layer) -> std::size_t {
    return (((aspect * k_extent) + mip) * k_extent) + layer;
  };
  std::array<cell_state, k_cell_count> cells{};
  std::array<std::array<cell_barrier, k_cell_count>, k_operation_count> expected{};
  std::array<vkexec::detail::usage_scope, k_operation_count> scopes{};
  std::array<vkexec::resource_access, k_operation_count> accesses{};
  std::set<std::pair<std::size_t, std::size_t>> edges;
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed) -- deterministic replay
  std::mt19937 random{ k_seed };
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_operation_count };
  for (std::size_t step{}; step < k_operation_count; ++step) {
    auto const aspect = static_cast<std::size_t>(random()) % k_aspects.size();
    auto const mip = static_cast<std::uint32_t>(random()) % k_extent;
    auto const layer = static_cast<std::uint32_t>(random()) % k_extent;
    auto const mips = k_one + (static_cast<std::uint32_t>(random()) % (k_extent - mip));
    auto const layers = k_one + (static_cast<std::uint32_t>(random()) % (k_extent - layer));
    auto const access = k_accesses.at(static_cast<std::size_t>(random()) % k_accesses.size());
    auto const usage = access == vkexec::resource_access::read && (random() & 1U) != 0
                         ? vkexec::image_usage::sampled_compute
                         : vkexec::image_usage::storage_compute;
    vkexec::image_use const use{ .image = image,
      .range = range(mip, mips, layer, layers, k_aspects.at(aspect)),
      .usage = usage,
      .access = access,
      .initial_layout = VK_IMAGE_LAYOUT_GENERAL };
    auto const scope = vkexec::detail::image_scope(use);
    scopes.at(step) = scope;
    accesses.at(step) = access;
    REQUIRE(tracker.use(use, step, k_family));
    for (auto current_mip = mip; current_mip < mip + mips; ++current_mip) {
      for (auto current_layer = layer; current_layer < layer + layers; ++current_layer) {
        auto const index = cell_index(aspect, current_mip, current_layer);
        auto &cell = cells.at(index);
        auto &barrier = expected.at(step).at(index);
        auto const synchronize = [&](std::size_t predecessor) -> void {
          edges.emplace(predecessor, step);
          barrier.source_stage |= scopes.at(predecessor).stage;
          barrier.source_access |= scopes.at(predecessor).access;
        };
        if (!cell.initialized) {
          barrier.present = true;
          if (access == vkexec::resource_access::read && scope.layout == cell.layout) {
            cell.readers.push_back(step);
          } else {
            cell.anchor = step;
          }
        } else if (access == vkexec::resource_access::read && cell.layout == scope.layout) {
          if (cell.anchor) {
            edges.emplace(*cell.anchor, step);
            if (accesses.at(*cell.anchor) != vkexec::resource_access::read) {
              barrier.present = true;
              synchronize(*cell.anchor);
            }
          }
          cell.readers.push_back(step);
        } else {
          barrier.present = true;
          if (!cell.readers.empty()) {
            for (auto const reader : cell.readers) { synchronize(reader); }
          } else if (cell.anchor) {
            synchronize(*cell.anchor);
          }
          cell.anchor = step;
          cell.readers.clear();
          if (access == vkexec::resource_access::read) { cell.readers.push_back(step); }
        }
        if (barrier.present) {
          barrier.old_layout = cell.layout;
          barrier.new_layout = scope.layout;
        }
        cell.layout = scope.layout;
        cell.initialized = true;
      }
    }
  }
  auto const plan = std::move(tracker).finish();
  std::set<std::pair<std::size_t, std::size_t>> const actual_edges{ plan.dependencies.begin(),
    plan.dependencies.end() };
  REQUIRE(actual_edges == edges);
  REQUIRE(actual_edges.size() == plan.dependencies.size());
  for (std::size_t step{}; step < k_operation_count; ++step) {
    INFO("step " << step << ", seed " << k_seed);
    std::array<cell_barrier, k_cell_count> actual{};
    for (auto const &barrier : plan.steps.at(step).images_before) {
      REQUIRE(barrier.image == image);
      REQUIRE(barrier.range.levelCount != 0);
      REQUIRE(barrier.range.layerCount != 0);
      REQUIRE(barrier.range.baseMipLevel + barrier.range.levelCount <= k_extent);
      REQUIRE(barrier.range.baseArrayLayer + barrier.range.layerCount <= k_extent);
      REQUIRE(barrier.src_queue_family == VK_QUEUE_FAMILY_IGNORED);
      REQUIRE(barrier.dst_queue_family == VK_QUEUE_FAMILY_IGNORED);
      REQUIRE(barrier.dst_stage == scopes.at(step).stage);
      REQUIRE(barrier.dst_access == scopes.at(step).access);
      for (std::size_t aspect{}; aspect < k_aspects.size(); ++aspect) {
        if ((barrier.range.aspectMask & k_aspects.at(aspect)) == 0) { continue; }
        for (auto mip = barrier.range.baseMipLevel; mip < barrier.range.baseMipLevel + barrier.range.levelCount;
          ++mip) {
          for (auto layer = barrier.range.baseArrayLayer;
            layer < barrier.range.baseArrayLayer + barrier.range.layerCount;
            ++layer) {
            auto &cell = actual.at(cell_index(aspect, mip, layer));
            REQUIRE_FALSE(cell.present);
            cell = cell_barrier{ .present = true,
              .old_layout = barrier.old_layout,
              .new_layout = barrier.new_layout,
              .source_stage = barrier.src_stage,
              .source_access = barrier.src_access };
          }
        }
      }
    }
    for (std::size_t index{}; index < k_cell_count; ++index) {
      INFO("cell " << index);
      auto const &observed = actual.at(index);
      auto const &wanted = expected.at(step).at(index);
      REQUIRE(observed.present == wanted.present);
      REQUIRE(observed.old_layout == wanted.old_layout);
      REQUIRE(observed.new_layout == wanted.new_layout);
      REQUIRE(observed.source_stage == wanted.source_stage);
      REQUIRE(observed.source_access == wanted.source_access);
    }
  }
}

TEST_CASE("image region merging keeps distinct reader frontiers and owners", "[vkexec][pass][ranges]")
{
  using vkexec::detail::image_region_state;
  using vkexec::detail::merge_image_regions;
  using vkexec::detail::normalize;
  using vkexec::detail::tracked_image;

  auto const first = normalize(range());
  auto const second = normalize(range(k_next));
  REQUIRE(first.has_value());
  REQUIRE(second.has_value());
  tracked_image const reader{ .access = vkexec::resource_access::read,
    .scope = { .stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
      .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
    .step = k_first_step,
    .family = k_family };
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access) -- Catch2's REQUIRE above checks engagement.
  std::vector<image_region_state> regions{
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access) -- Catch2's REQUIRE above checks engagement.
    image_region_state{ .region = *first,
      .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .family = k_family,
      .accesses = { .anchor = std::nullopt, .readers = { reader } } },
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access) -- Catch2's REQUIRE above checks engagement.
    image_region_state{ .region = *second,
      .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .family = k_family,
      .accesses = { .anchor = std::nullopt, .readers = { reader } } }
  };
  SECTION("different reader step") { regions.back().accesses.readers.front().step = k_second_step; }
  SECTION("different stage")
  { regions.back().accesses.readers.front().scope.stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT; }
  SECTION("different access mask")
  { regions.back().accesses.readers.front().scope.access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT; }
  SECTION("different owner") { regions.back().family = k_other_family; }
  SECTION("different layout") { regions.back().layout = VK_IMAGE_LAYOUT_GENERAL; }
  SECTION("different anchor") { regions.back().accesses.anchor = reader; }
  merge_image_regions(regions);
  REQUIRE(regions.size() == k_two_barriers);
}

TEST_CASE("registered image extent eliminates empty remaining-range tails", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  auto const remaining = range(k_base, VK_REMAINING_MIP_LEVELS, k_base, VK_REMAINING_ARRAY_LAYERS);
  REQUIRE(
    tracker.use(vkexec::uses(vkexec::image_extent{ .image = image, .mip_levels = k_extent, .array_layers = k_extent },
                  vkexec::write(image, remaining, vkexec::image_usage::storage_compute)),
      k_first_step,
      k_family));
  REQUIRE(
    tracker.use(vkexec::read(image, range(k_base, k_extent, k_base, k_extent), vkexec::image_usage::sampled_compute),
      k_second_step,
      k_family));
  REQUIRE(tracker.use(vkexec::write(image, remaining, vkexec::image_usage::storage_compute), k_third_step, k_family));
  REQUIRE(tracker.region_counts().images == k_one_barrier);
  REQUIRE_FALSE(
    tracker.use(vkexec::read(image, range(k_extent), vkexec::image_usage::sampled_compute), k_fourth_step, k_family));
  REQUIRE_FALSE(tracker.use(vkexec::read(image, range(k_base, k_one, k_extent), vkexec::image_usage::sampled_compute),
    k_fourth_step,
    k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_third_step).images_before.size() == k_one_barrier);
  require_range(plan.steps.at(k_third_step).images_before.front().range, range(k_base, k_extent, k_base, k_extent));
  REQUIRE(plan.steps.at(k_fourth_step).images_before.empty());
}

TEST_CASE("concurrent images retain layout dependencies without ownership transfers", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(tracker.register_extent(vkexec::image_extent{
    .image = image, .mip_levels = k_extent, .array_layers = k_one, .sharing_mode = VK_SHARING_MODE_CONCURRENT }));
  REQUIRE(tracker.use(
    vkexec::write(image, range(k_base, k_extent), vkexec::image_usage::storage_compute), k_first_step, k_family));
  REQUIRE(tracker.use(
    vkexec::read(image, range(k_next), vkexec::image_usage::sampled_compute), k_second_step, k_other_family));
  REQUIRE(
    tracker.use(vkexec::read(image, range(k_next), vkexec::image_usage::sampled_fragment), k_third_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(has_edge(plan, k_first_step, k_second_step));
  REQUIRE(has_edge(plan, k_second_step, k_third_step));
  REQUIRE(plan.steps.at(k_second_step).images_before.front().old_layout == VK_IMAGE_LAYOUT_GENERAL);
  REQUIRE(plan.steps.at(k_second_step).images_before.front().new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  for (auto const &step : plan.steps) {
    REQUIRE(step.images_after.empty());
    for (auto const &barrier : step.images_before) {
      REQUIRE(barrier.src_queue_family == VK_QUEUE_FAMILY_IGNORED);
      REQUIRE(barrier.dst_queue_family == VK_QUEUE_FAMILY_IGNORED);
    }
  }
}

#if defined(VKEXEC_ENABLE_ALLOCATION_FAULT_INJECTION) && VKEXEC_HAS_EXCEPTIONS
// VKEXEC_HAS_EXCEPTIONS comes from <vkexec/config.hpp> (compiler unwind / __cpp_exceptions).
TEST_CASE("uses batch rolls back when image barrier generation throws before insert", "[vkexec][pass][ranges]")
{
  char existing_storage{};
  char created_storage{};
  auto *const existing = fake_image(&existing_storage);
  auto *const created = fake_image(&created_storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(
    tracker.use(vkexec::write(existing, range(k_base), vkexec::image_usage::storage_compute), k_first_step, k_family));
  vkexec::detail::resource_allocation_fault::arm(1);
  REQUIRE_THROWS_AS(
    tracker.use(vkexec::uses(vkexec::write(created, range(k_base), vkexec::image_usage::storage_compute)),
      k_second_step,
      k_family),
    std::bad_alloc);
  vkexec::detail::resource_allocation_fault::disarm();
  REQUIRE(tracker.region_counts().images == k_one_barrier);
  REQUIRE(
    tracker.use(vkexec::write(created, range(k_base), vkexec::image_usage::storage_compute), k_second_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_first_step).images_before.size() == k_one_barrier);
  REQUIRE(plan.steps.at(k_second_step).images_before.size() == k_one_barrier);
  REQUIRE(plan.dependencies.empty());
}

TEST_CASE("uses batch rolls back when image insertion throws after barriers", "[vkexec][pass][ranges]")
{
  char existing_storage{};
  char created_storage{};
  auto *const existing = fake_image(&existing_storage);
  auto *const created = fake_image(&created_storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(
    tracker.use(vkexec::write(existing, range(k_base), vkexec::image_usage::storage_compute), k_first_step, k_family));
  // First checkpoint allows barrier generation.
  // Second checkpoint throws before tracked-image insertion.
  vkexec::detail::resource_allocation_fault::arm(2);
  REQUIRE_THROWS_AS(
    tracker.use(vkexec::uses(vkexec::write(created, range(k_base), vkexec::image_usage::storage_compute)),
      k_second_step,
      k_family),
    std::bad_alloc);
  REQUIRE(vkexec::detail::resource_allocation_fault::remaining == 0);
  vkexec::detail::resource_allocation_fault::disarm();
  REQUIRE(tracker.region_counts().images == k_one_barrier);
  REQUIRE(
    tracker.use(vkexec::write(created, range(k_base), vkexec::image_usage::storage_compute), k_second_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_first_step).images_before.size() == k_one_barrier);
  REQUIRE(plan.steps.at(k_second_step).images_before.size() == k_one_barrier);
  REQUIRE(plan.image_uses.size() == k_two_barriers);
}

#endif

// NOLINTEND(readability-function-cognitive-complexity)
