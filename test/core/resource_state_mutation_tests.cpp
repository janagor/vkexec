#include "pass_detail_test_support.hpp"

#include <catch2/catch_message.hpp>

TEST_CASE("buffer access matrix preserves exact hazard edges and scopes", "[vkexec][resource-state]")
{
  using enum vkexec::resource_access;
  std::vector<std::vector<vkexec::resource_access>> const sequences{ { read, read },
    { write, read },
    { read, write },
    { write, write },
    { read, read, write },
    { write, read, read },
    { write, read, write } };
  std::vector<std::vector<std::pair<std::size_t, std::size_t>>> const edges{
    {}, { { 0, 1 } }, { { 0, 1 } }, { { 0, 1 } }, { { 0, 2 }, { 1, 2 } }, { { 0, 1 }, { 0, 2 } }, { { 0, 1 }, { 1, 2 } }
  };
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  for (std::size_t index{}; index < sequences.size(); ++index) {
    CAPTURE(index);
    auto const &sequence = sequences.at(index);
    vkexec::detail::resource_state_tracker tracker{ sequence.size() };
    for (std::size_t step{}; step < sequence.size(); ++step) {
      REQUIRE(tracker.use(vkexec::buffer_use{ .buffer = buffer,
                            .offset = 0,
                            .size = k_byte_size,
                            .usage = vkexec::buffer_usage::storage_compute,
                            .access = sequence.at(step) },
        step,
        k_default_family));
    }
    auto const plan = std::move(tracker).finish();
    REQUIRE(plan.dependencies == edges.at(index));
    for (std::size_t step{}; step < sequence.size(); ++step) {
      auto const &point = plan.steps.at(step);
      REQUIRE(point.buffers_after.empty());
      bool const hazard = step != 0 && (sequence.at(step) == write || sequence.front() == write);
      REQUIRE(point.buffers_before.size() == (hazard ? 1 : 0));
      if (!hazard) { continue; }
      auto const &barrier = point.buffers_before.front();
      REQUIRE(barrier.buffer == buffer);
      REQUIRE(barrier.offset == 0);
      REQUIRE(barrier.size == k_byte_size);
      REQUIRE(barrier.src_stage == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
      REQUIRE(barrier.dst_stage == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
      REQUIRE(barrier.src_access
              == ((sequence.at(step) == read ? sequence.front() : sequence.at(step - 1)) == write
                    ? VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
                    : VK_ACCESS_2_SHADER_STORAGE_READ_BIT));
      REQUIRE(
        barrier.dst_access
        == (sequence.at(step) == write ? VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT : VK_ACCESS_2_SHADER_STORAGE_READ_BIT));
      REQUIRE(barrier.src_queue_family == VK_QUEUE_FAMILY_IGNORED);
      REQUIRE(barrier.dst_queue_family == VK_QUEUE_FAMILY_IGNORED);
    }
  }
}

TEST_CASE("extent registration validates identity dimensions and sharing consistently", "[vkexec][resource-state]")
{
  char buffer_storage{};
  char image_storage{};
  auto *const buffer = fake_buffer(&buffer_storage);
  auto *const image = fake_image(&image_storage);
  vkexec::detail::resource_state_tracker tracker{ k_pair_step_count };
  vkexec::buffer_extent buffer_extent{ .buffer = buffer, .size = k_byte_size };
  vkexec::image_extent image_extent{ .image = image, .mip_levels = 2, .array_layers = 3 };
  REQUIRE_FALSE(tracker.register_extent(vkexec::buffer_extent{}));
  REQUIRE_FALSE(tracker.register_extent(vkexec::image_extent{}));
  for (auto const size : { VkDeviceSize{ 0 }, k_byte_size }) {
    auto invalid = buffer_extent;
    invalid.size = size;
    invalid.buffer = size == 0 ? buffer : VK_NULL_HANDLE;
    REQUIRE_FALSE(tracker.register_extent(invalid));
  }
  for (auto const count : { 0U, VK_REMAINING_MIP_LEVELS }) {
    auto invalid = image_extent;
    invalid.mip_levels = count;
    REQUIRE_FALSE(tracker.register_extent(invalid));
    invalid = image_extent;
    invalid.array_layers = count;
    REQUIRE_FALSE(tracker.register_extent(invalid));
  }
  REQUIRE(tracker.register_extent(buffer_extent));
  REQUIRE(tracker.register_extent(buffer_extent));
  REQUIRE(tracker.register_extent(image_extent));
  REQUIRE(tracker.register_extent(image_extent));
  auto changed_buffer = buffer_extent;
  changed_buffer.size += 1;
  REQUIRE_FALSE(tracker.register_extent(changed_buffer));
  changed_buffer = buffer_extent;
  changed_buffer.sharing_mode = VK_SHARING_MODE_CONCURRENT;
  REQUIRE_FALSE(tracker.register_extent(changed_buffer));
  auto changed_image = image_extent;
  changed_image.mip_levels += 1;
  REQUIRE_FALSE(tracker.register_extent(changed_image));
  changed_image = image_extent;
  changed_image.array_layers += 1;
  REQUIRE_FALSE(tracker.register_extent(changed_image));
  changed_image = image_extent;
  changed_image.sharing_mode = VK_SHARING_MODE_CONCURRENT;
  REQUIRE_FALSE(tracker.register_extent(changed_image));
  REQUIRE_FALSE(
    tracker.use(vkexec::read(buffer, vkexec::buffer_usage::storage_compute), k_pair_step_count, k_default_family));
  REQUIRE_FALSE(tracker.use(
    vkexec::write(image, k_color_range, vkexec::image_usage::storage_compute), k_pair_step_count, k_default_family));
  REQUIRE_FALSE(
    tracker.use(vkexec::read(VkBuffer{}, vkexec::buffer_usage::storage_compute), k_first_step, k_default_family));
  REQUIRE_FALSE(tracker.use(
    vkexec::write(VkImage{}, k_color_range, vkexec::image_usage::storage_compute), k_first_step, k_default_family));
}

namespace {
auto range_key(VkImageSubresourceRange const &range)
{ return std::tuple{ range.aspectMask, range.baseMipLevel, range.levelCount, range.baseArrayLayer, range.layerCount }; }
auto barrier_key(vkexec::buffer_barrier_params const &barrier)
{
  return std::tuple{ barrier.buffer,
    barrier.offset,
    barrier.size,
    barrier.src_stage,
    barrier.dst_stage,
    barrier.src_access,
    barrier.dst_access,
    barrier.src_queue_family,
    barrier.dst_queue_family };
}
auto barrier_key(vkexec::image_barrier_params const &barrier)
{
  return std::tuple{ barrier.image,
    range_key(barrier.range),
    barrier.old_layout,
    barrier.new_layout,
    barrier.src_stage,
    barrier.dst_stage,
    barrier.src_access,
    barrier.dst_access,
    barrier.src_queue_family,
    barrier.dst_queue_family };
}
template<class Barrier>
auto require_barriers(std::vector<Barrier> const &actual, std::vector<Barrier> const &expected) -> void
{
  REQUIRE(actual.size() == expected.size());
  for (std::size_t index{}; index < actual.size(); ++index) {
    REQUIRE(barrier_key(actual.at(index)) == barrier_key(expected.at(index)));
  }
}
auto require_plan(vkexec::detail::resource_sync_plan const &actual, vkexec::detail::resource_sync_plan const &expected)
  -> void
{
  REQUIRE(actual.dependencies == expected.dependencies);
  REQUIRE(actual.steps.size() == expected.steps.size());
  REQUIRE(actual.image_uses.size() == expected.image_uses.size());
  for (std::size_t index{}; index < actual.steps.size(); ++index) {
    auto const &left = actual.steps.at(index);
    auto const &right = expected.steps.at(index);
    require_barriers(left.buffers_before, right.buffers_before);
    require_barriers(left.buffers_after, right.buffers_after);
    require_barriers(left.images_before, right.images_before);
    require_barriers(left.images_after, right.images_after);
  }
  for (std::size_t index{}; index < actual.image_uses.size(); ++index) {
    auto const &left = actual.image_uses.at(index);
    auto const &right = expected.image_uses.at(index);
    REQUIRE(std::tuple{ left.image, left.usage, left.final_layout, left.step, range_key(left.range) }
            == std::tuple{ right.image, right.usage, right.final_layout, right.step, range_key(right.range) });
  }
}
}// namespace

TEST_CASE("failed mixed batch restores the complete synchronization plan", "[vkexec][resource-state]")
{
  char buffer_storage{};
  char image_storage{};
  char new_buffer_storage{};
  char new_image_storage{};
  auto *const buffer = fake_buffer(&buffer_storage);
  auto *const image = fake_image(&image_storage);
  auto *const new_buffer = fake_buffer(&new_buffer_storage);
  auto *const new_image = fake_image(&new_image_storage);
  for (bool const fail_image : { false, true }) {
    for (bool const introduce : { false, true }) {
      CAPTURE(fail_image, introduce);
      auto run = [&](bool fail_batch) {
        vkexec::detail::resource_state_tracker tracker{ 3 };
        REQUIRE(tracker.use(vkexec::uses(vkexec::write(buffer, 0, k_byte_size, vkexec::buffer_usage::storage_compute),
                              vkexec::write(image, k_color_range, vkexec::image_usage::storage_compute)),
          0,
          k_default_family));
        if (fail_batch) {
          auto const buffer_use =
            vkexec::read(introduce ? new_buffer : buffer, 0, k_byte_size, vkexec::buffer_usage::storage_compute);
          auto const image_use =
            vkexec::write(introduce ? new_image : image, k_color_range, vkexec::image_usage::storage_compute);
          if (fail_image) {
            REQUIRE_FALSE(
              tracker.use(vkexec::uses(image_use,
                            vkexec::read(image_use.image, k_color_range, vkexec::image_usage::transfer_destination)),
                1,
                k_other_family));
          } else {
            REQUIRE_FALSE(
              tracker.use(vkexec::uses(image_use,
                            buffer_use,
                            vkexec::write(buffer_use.buffer, 0, k_byte_size, vkexec::buffer_usage::uniform_compute)),
                1,
                k_other_family));
          }
        }
        REQUIRE(tracker.region_counts().buffers == 1);
        REQUIRE(tracker.region_counts().images == 1);
        // Different extents expose leaked registrations even when they contain no regions.
        REQUIRE(tracker.register_extent(vkexec::buffer_extent{ .buffer = new_buffer, .size = k_byte_size * 2 }));
        REQUIRE(
          tracker.register_extent(vkexec::image_extent{ .image = new_image, .mip_levels = 2, .array_layers = 2 }));
        REQUIRE(tracker.use(vkexec::uses(vkexec::read(buffer, 0, k_byte_size, vkexec::buffer_usage::uniform_compute),
                              vkexec::read(image, k_color_range, vkexec::image_usage::sampled_compute)),
          2,
          k_other_family));
        return std::move(tracker).finish();
      };
      require_plan(run(true), run(false));
    }
  }
}

TEST_CASE("ownership matrix emits precise buffer and image handoff scopes", "[vkexec][resource-state]")
{
  char buffer_storage{};
  char image_storage{};
  char first_queue_storage{};
  char second_queue_storage{};
  auto *const buffer = fake_buffer(&buffer_storage);
  auto *const image = fake_image(&image_storage);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only handle
  auto *const first_queue = reinterpret_cast<VkQueue>(&first_queue_storage);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only handle
  auto *const second_queue = reinterpret_cast<VkQueue>(&second_queue_storage);
  for (auto const sharing : { VK_SHARING_MODE_EXCLUSIVE, VK_SHARING_MODE_CONCURRENT }) {
    for (bool const cross_family : { false, true }) {
      for (bool const cross_queue : { false, true }) {
        if (cross_family && !cross_queue) { continue; }
        CAPTURE(sharing, cross_family, cross_queue);
        auto const family = cross_family ? k_other_family : k_default_family;
        auto *const queue = cross_queue ? second_queue : first_queue;
        vkexec::detail::resource_state_tracker tracker{ 2 };
        REQUIRE(tracker.register_extent(
          vkexec::buffer_extent{ .buffer = buffer, .size = k_byte_size, .sharing_mode = sharing }));
        REQUIRE(tracker.register_extent(
          vkexec::image_extent{ .image = image, .mip_levels = 1, .array_layers = 1, .sharing_mode = sharing }));
        REQUIRE(tracker.use(vkexec::uses(vkexec::write(buffer, vkexec::buffer_usage::storage_compute),
                              vkexec::write(image, k_color_range, vkexec::image_usage::storage_compute)),
          0,
          k_default_family,
          first_queue));
        REQUIRE(tracker.use(vkexec::uses(vkexec::read(buffer, vkexec::buffer_usage::uniform_compute),
                              vkexec::read(image, k_color_range, vkexec::image_usage::sampled_compute)),
          1,
          family,
          queue));
        auto const plan = std::move(tracker).finish();
        REQUIRE(plan.dependencies == std::vector<std::pair<std::size_t, std::size_t>>{ { 0, 1 } });
        bool const transfer = cross_family && sharing == VK_SHARING_MODE_EXCLUSIVE;
        auto const &source = plan.steps.at(0);
        auto const &destination = plan.steps.at(1);
        REQUIRE(source.buffers_after.size() == (transfer ? 1 : 0));
        REQUIRE(source.images_after.size() == (transfer ? 1 : 0));
        REQUIRE(destination.buffers_before.size() == 1);
        REQUIRE(destination.images_before.size() == 1);
        auto const &acquire_buffer = destination.buffers_before.front();
        auto const &acquire_image = destination.images_before.front();
        for (auto const &barrier : { acquire_buffer }) {
          REQUIRE(
            barrier.src_stage == (cross_queue ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT));
          REQUIRE(barrier.src_access == (cross_queue ? VK_ACCESS_2_NONE : VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT));
          REQUIRE(barrier.dst_stage == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
          REQUIRE(barrier.dst_access == VK_ACCESS_2_UNIFORM_READ_BIT);
          REQUIRE(barrier.src_queue_family == (transfer ? k_default_family : VK_QUEUE_FAMILY_IGNORED));
          REQUIRE(barrier.dst_queue_family == (transfer ? k_other_family : VK_QUEUE_FAMILY_IGNORED));
          REQUIRE(barrier.offset == 0);
          REQUIRE(barrier.size == k_byte_size);
        }
        REQUIRE(acquire_image.src_stage == acquire_buffer.src_stage);
        REQUIRE(acquire_image.src_access == acquire_buffer.src_access);
        REQUIRE(acquire_image.dst_stage == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        REQUIRE(acquire_image.dst_access == VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        REQUIRE(acquire_image.src_queue_family == acquire_buffer.src_queue_family);
        REQUIRE(acquire_image.dst_queue_family == acquire_buffer.dst_queue_family);
        REQUIRE(acquire_image.old_layout == VK_IMAGE_LAYOUT_GENERAL);
        REQUIRE(acquire_image.new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (transfer) {
          auto const &release_buffer = source.buffers_after.front();
          auto const &release_image = source.images_after.front();
          REQUIRE(release_buffer.src_stage == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
          REQUIRE(release_buffer.src_access == VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
          REQUIRE(release_buffer.dst_stage == VK_PIPELINE_STAGE_2_NONE);
          REQUIRE(release_buffer.dst_access == VK_ACCESS_2_NONE);
          REQUIRE(release_buffer.src_queue_family == acquire_buffer.src_queue_family);
          REQUIRE(release_buffer.dst_queue_family == acquire_buffer.dst_queue_family);
          REQUIRE(release_image.src_stage == release_buffer.src_stage);
          REQUIRE(release_image.src_access == release_buffer.src_access);
          REQUIRE(release_image.dst_stage == VK_PIPELINE_STAGE_2_NONE);
          REQUIRE(release_image.dst_access == VK_ACCESS_2_NONE);
          REQUIRE(release_image.src_queue_family == acquire_image.src_queue_family);
          REQUIRE(release_image.dst_queue_family == acquire_image.dst_queue_family);
          REQUIRE(release_image.old_layout == acquire_image.old_layout);
          REQUIRE(release_image.new_layout == acquire_image.new_layout);
          REQUIRE(range_key(release_image.range) == range_key(acquire_image.range));
        }
      }
    }
  }
}

TEST_CASE("rollback journals retain the earliest checkpoint of each resource", "[vkexec][resource-state]")
{
  char buffer_storage{};
  char image_storage{};
  auto *const buffer = fake_buffer(&buffer_storage);
  auto *const image = fake_image(&image_storage);
  auto run = [&](bool failed) {
    vkexec::detail::resource_state_tracker tracker{ 3 };
    REQUIRE(tracker.use(vkexec::uses(vkexec::write(buffer, 0, k_byte_size, vkexec::buffer_usage::storage_compute),
                          vkexec::write(image, k_color_range, vkexec::image_usage::storage_compute)),
      0,
      k_default_family));
    if (failed) {
      // Two compatible reads journal the same pre-existing resource twice.
      REQUIRE_FALSE(tracker.use(vkexec::uses(vkexec::read(image, k_color_range, vkexec::image_usage::sampled_compute),
                                  vkexec::read(image, k_color_range, vkexec::image_usage::sampled_compute),
                                  vkexec::read(buffer, 0, k_byte_size, vkexec::buffer_usage::uniform_compute),
                                  vkexec::read(buffer, 0, k_byte_size, vkexec::buffer_usage::uniform_compute),
                                  vkexec::write(buffer, 0, k_byte_size, vkexec::buffer_usage::uniform_compute)),
        1,
        k_default_family));
    }
    REQUIRE(tracker.use(vkexec::uses(vkexec::write(buffer, 0, k_byte_size, vkexec::buffer_usage::storage_compute),
                          vkexec::read(image, k_color_range, vkexec::image_usage::sampled_compute)),
      2,
      k_default_family));
    REQUIRE(tracker.region_counts().buffers == 1);
    REQUIRE(tracker.region_counts().images == 1);
    return std::move(tracker).finish();
  };
  require_plan(run(true), run(false));
}

#if defined(VKEXEC_ENABLE_ALLOCATION_FAULT_INJECTION) && VKEXEC_HAS_EXCEPTIONS
TEST_CASE("allocation checkpoints restore a mixed batch and permit retry",
  "[vkexec][resource-state][resource-state-fault]")
{
  char buffer_storage{};
  char image_storage{};
  auto *const buffer = fake_buffer(&buffer_storage);
  auto *const image = fake_image(&image_storage);
  for (int checkpoint = 1; checkpoint <= 3; ++checkpoint) {
    CAPTURE(checkpoint);
    auto run = [&](bool inject) {
      vkexec::detail::resource_state_tracker tracker{ 3 };
      if (inject) {
        struct fault_cleanup
        {
          ~fault_cleanup() { vkexec::detail::resource_allocation_fault::disarm(); }
        } const cleanup;
        vkexec::detail::resource_allocation_fault::arm(checkpoint);
        REQUIRE_THROWS_AS(
          tracker.use(vkexec::uses(vkexec::image_extent{ .image = image, .mip_levels = 1, .array_layers = 1 },
                        vkexec::buffer_extent{ .buffer = buffer, .size = k_byte_size },
                        vkexec::write(image, k_color_range, vkexec::image_usage::storage_compute),
                        vkexec::write(buffer, 0, k_byte_size, vkexec::buffer_usage::storage_compute)),
            0,
            k_default_family),
          std::bad_alloc);
      }
      REQUIRE(tracker.region_counts().buffers == 0);
      REQUIRE(tracker.region_counts().images == 0);
      REQUIRE(tracker.register_extent(vkexec::buffer_extent{ .buffer = buffer, .size = k_byte_size * 2 }));
      REQUIRE(tracker.register_extent(vkexec::image_extent{ .image = image, .mip_levels = 2, .array_layers = 2 }));
      REQUIRE(tracker.use(vkexec::uses(vkexec::write(image, k_color_range, vkexec::image_usage::storage_compute),
                            vkexec::write(buffer, 0, k_byte_size, vkexec::buffer_usage::storage_compute)),
        1,
        k_default_family));
      REQUIRE(tracker.use(vkexec::uses(vkexec::read(image, k_color_range, vkexec::image_usage::sampled_compute),
                            vkexec::read(buffer, 0, k_byte_size, vkexec::buffer_usage::uniform_compute)),
        2,
        k_other_family));
      return std::move(tracker).finish();
    };
    require_plan(run(true), run(false));
  }
}
#endif

TEST_CASE("attachment initial and final layouts carry into later reads", "[vkexec][resource-state]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  for (auto const initial : { VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL }) {
    CAPTURE(initial);
    vkexec::detail::resource_state_tracker tracker{ 3 };
    auto attachment = vkexec::write(image, k_color_range, vkexec::image_usage::color_attachment);
    attachment.final_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    REQUIRE(tracker.use_attachment(attachment, initial, 0, k_default_family));
    REQUIRE(tracker.use(vkexec::read(image, k_color_range, vkexec::image_usage::sampled_compute), 1, k_default_family));
    REQUIRE(
      tracker.use(vkexec::read(image, k_color_range, vkexec::image_usage::sampled_fragment), 2, k_default_family));
    auto const plan = std::move(tracker).finish();
    REQUIRE(plan.dependencies == std::vector<std::pair<std::size_t, std::size_t>>{ { 0, 1 }, { 0, 2 } });
    REQUIRE(plan.steps.at(0).images_before.size() == (initial == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : 1));
    if (initial != VK_IMAGE_LAYOUT_UNDEFINED) {
      auto const &barrier = plan.steps.at(0).images_before.front();
      REQUIRE(barrier.old_layout == VK_IMAGE_LAYOUT_UNDEFINED);
      REQUIRE(barrier.new_layout == initial);
      REQUIRE(barrier.src_stage == VK_PIPELINE_STAGE_2_NONE);
      REQUIRE(barrier.src_access == VK_ACCESS_2_NONE);
      REQUIRE(barrier.dst_stage == VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
      REQUIRE(barrier.dst_access == VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    }
    for (std::size_t step = 1; step < 3; ++step) {
      REQUIRE(plan.steps.at(step).images_before.size() == 1);
      auto const &barrier = plan.steps.at(step).images_before.front();
      REQUIRE(barrier.old_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      REQUIRE(barrier.new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      REQUIRE(barrier.src_stage == VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
      REQUIRE(barrier.src_access == VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      REQUIRE(barrier.dst_stage
              == (step == 1 ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT));
      REQUIRE(barrier.dst_access == VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
  }
}

TEST_CASE("rollback restores distinct resources with separate journal entries", "[vkexec][resource-state]")
{
  std::array<char, 2> buffers{};
  std::array<char, 2> images{};
  auto run = [&](bool failed) {
    vkexec::detail::resource_state_tracker tracker{ 3 };
    for (std::size_t index{}; index < buffers.size(); ++index) {
      REQUIRE(tracker.use(
        vkexec::uses(
          vkexec::write(fake_buffer(&buffers.at(index)), 0, k_byte_size, vkexec::buffer_usage::storage_compute),
          vkexec::write(fake_image(&images.at(index)), k_color_range, vkexec::image_usage::storage_compute)),
        0,
        k_default_family));
    }
    if (failed) {
      REQUIRE_FALSE(tracker.use(
        vkexec::uses(vkexec::read(fake_image(&images.at(0)), k_color_range, vkexec::image_usage::sampled_compute),
          vkexec::read(fake_image(&images.at(1)), k_color_range, vkexec::image_usage::sampled_compute),
          vkexec::read(fake_buffer(&buffers.at(0)), 0, k_byte_size, vkexec::buffer_usage::uniform_compute),
          vkexec::read(fake_buffer(&buffers.at(1)), 0, k_byte_size, vkexec::buffer_usage::uniform_compute),
          vkexec::write(fake_buffer(&buffers.at(1)), 0, k_byte_size, vkexec::buffer_usage::uniform_compute)),
        1,
        k_other_family));
    }
    for (std::size_t index{}; index < buffers.size(); ++index) {
      REQUIRE(tracker.use(
        vkexec::uses(
          vkexec::read(fake_buffer(&buffers.at(index)), 0, k_byte_size, vkexec::buffer_usage::uniform_compute),
          vkexec::read(fake_image(&images.at(index)), k_color_range, vkexec::image_usage::sampled_compute)),
        2,
        k_default_family));
    }
    REQUIRE(tracker.region_counts().buffers == 2);
    REQUIRE(tracker.region_counts().images == 2);
    return std::move(tracker).finish();
  };
  require_plan(run(true), run(false));
}
