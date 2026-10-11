#include <vkexec/detail/resource_state_tracker.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/result.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <vulkan/vulkan_core.h>

namespace {

constexpr std::size_t k_buffer_steps = 512;
constexpr VkDeviceSize k_chunk_size = 64;
constexpr VkDeviceSize k_overlap_stride = 16;
constexpr std::size_t k_overlap_positions = 32;
constexpr std::uint32_t k_mip_count = 12;
constexpr std::uint32_t k_layer_count = 256;
constexpr std::uint32_t k_family = 0;
constexpr std::uint32_t k_single_level = 1;
constexpr std::uint32_t k_single_layer = 1;
constexpr std::uint32_t k_base_mip = 0;
constexpr std::uint32_t k_base_layer = 0;
constexpr VkDeviceSize k_identical_offset = 0;

[[nodiscard]] constexpr auto range(std::uint32_t mip, std::uint32_t layer) -> VkImageSubresourceRange
{
  return { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = mip,
    .levelCount = k_single_level,
    .baseArrayLayer = layer,
    .layerCount = k_single_layer };
}

[[nodiscard]] auto barrier_count(vkexec::detail::resource_sync_plan const &plan) -> std::size_t
{
  std::size_t barriers{};
  for (auto const &point : plan.steps) {
    barriers +=
      point.buffers_before.size() + point.buffers_after.size() + point.images_before.size() + point.images_after.size();
  }
  return barriers;
}

[[nodiscard]] auto opaque_buffer() -> VkBuffer
{
  static char storage{};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only buffer handle
  return reinterpret_cast<VkBuffer>(&storage);
}

[[nodiscard]] auto opaque_image() -> VkImage
{
  static char storage{};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only image handle
  return reinterpret_cast<VkImage>(&storage);
}

auto set_plan_counters(benchmark::State &state,
  std::size_t peak_regions,
  std::size_t final_regions,
  std::size_t barriers,
  std::size_t dependencies) -> void
{
  state.counters.insert_or_assign("peak_regions", static_cast<double>(peak_regions));
  state.counters.insert_or_assign("final_regions", static_cast<double>(final_regions));
  state.counters.insert_or_assign("barriers", static_cast<double>(barriers));
  state.counters.insert_or_assign("dependencies", static_cast<double>(dependencies));
}

template<class Declare> auto run_planning(benchmark::State &state, std::size_t steps, Declare declare) -> void
{
  std::size_t peak_regions{};
  std::size_t final_regions{};
  std::size_t barriers{};
  std::size_t dependencies{};

  // NOLINTNEXTLINE(readability-identifier-length) -- Google Benchmark keep-running idiom
  for (auto _ : state) {
    peak_regions = 0;
    vkexec::detail::resource_state_tracker tracker{ steps };
    for (std::size_t step{}; step < steps; ++step) {
      auto const result = declare(tracker, step);
      if (!result) {
        state.SkipWithError(result.error().message());
        return;
      }
      auto const counts = tracker.region_counts();
      final_regions = counts.buffers + counts.images;
      peak_regions = std::max(peak_regions, final_regions);
    }
    auto const plan = std::move(tracker).finish();
    dependencies = plan.dependencies.size();
    barriers = barrier_count(plan);
  }

  set_plan_counters(state, peak_regions, final_regions, barriers, dependencies);
}

auto adjacent_buffer_writes(benchmark::State &state) -> void
{
  auto *const buffer = opaque_buffer();
  auto const steps = static_cast<std::size_t>(state.range(0));
  run_planning(
    state, steps, [buffer](vkexec::detail::resource_state_tracker &tracker, std::size_t step) -> vkexec::status {
      return tracker.use(
        vkexec::write(
          buffer, static_cast<VkDeviceSize>(step) * k_chunk_size, k_chunk_size, vkexec::buffer_usage::storage_compute),
        step,
        k_family);
    });
}

auto alternating_buffer_overlaps(benchmark::State &state) -> void
{
  auto *const buffer = opaque_buffer();
  auto const steps = static_cast<std::size_t>(state.range(0));
  run_planning(
    state, steps, [buffer](vkexec::detail::resource_state_tracker &tracker, std::size_t step) -> vkexec::status {
      vkexec::buffer_use const use{ .buffer = buffer,
        .offset = static_cast<VkDeviceSize>(step % k_overlap_positions) * k_overlap_stride,
        .size = k_chunk_size,
        .usage = vkexec::buffer_usage::storage_compute,
        .access = (step & 1U) == 0 ? vkexec::resource_access::write : vkexec::resource_access::read };
      return tracker.use(use, step, k_family);
    });
}

auto identical_buffer_ranges(benchmark::State &state) -> void
{
  auto *const buffer = opaque_buffer();
  auto const steps = static_cast<std::size_t>(state.range(0));
  run_planning(
    state, steps, [buffer](vkexec::detail::resource_state_tracker &tracker, std::size_t step) -> vkexec::status {
      return tracker.use(
        vkexec::write(buffer, k_identical_offset, k_chunk_size, vkexec::buffer_usage::storage_compute), step, k_family);
    });
}

auto mipmap_chain(benchmark::State &state) -> void
{
  auto *const image = opaque_image();
  auto const steps = static_cast<std::size_t>(state.range(0));
  run_planning(
    state, steps, [image](vkexec::detail::resource_state_tracker &tracker, std::size_t step) -> vkexec::status {
      auto const mip = static_cast<std::uint32_t>(step);
      if (mip != k_base_mip) {
        VKEXEC_TRY(tracker.use(
          vkexec::read(image, range(mip - k_single_level, k_base_layer), vkexec::image_usage::transfer_source),
          step,
          k_family));
      }
      return tracker.use(
        vkexec::write(image, range(mip, k_base_layer), vkexec::image_usage::transfer_destination), step, k_family);
    });
}

auto individual_image_layers(benchmark::State &state) -> void
{
  auto *const image = opaque_image();
  auto const steps = static_cast<std::size_t>(state.range(0));
  run_planning(
    state, steps, [image](vkexec::detail::resource_state_tracker &tracker, std::size_t step) -> vkexec::status {
      return tracker.use(
        vkexec::write(image, range(k_base_mip, static_cast<std::uint32_t>(step)), vkexec::image_usage::storage_compute),
        step,
        k_family);
    });
}

auto multipass_uses_batches(benchmark::State &state) -> void
{
  auto *const buffer = opaque_buffer();
  auto const steps = static_cast<std::size_t>(state.range(0));
  run_planning(
    state, steps, [buffer](vkexec::detail::resource_state_tracker &tracker, std::size_t step) -> vkexec::status {
      auto const write_offset = static_cast<VkDeviceSize>(step) * (k_chunk_size + k_chunk_size);
      auto const read_offset = write_offset + k_chunk_size;
      return tracker.use(
        vkexec::uses(vkexec::write(buffer, write_offset, k_chunk_size, vkexec::buffer_usage::storage_compute),
          vkexec::read(buffer, read_offset, k_chunk_size, vkexec::buffer_usage::uniform_compute)),
        step,
        k_family);
    });
}

BENCHMARK(adjacent_buffer_writes)->Unit(benchmark::kMicrosecond)->Arg(static_cast<std::int64_t>(k_buffer_steps));
BENCHMARK(alternating_buffer_overlaps)->Unit(benchmark::kMicrosecond)->Arg(static_cast<std::int64_t>(k_buffer_steps));
BENCHMARK(identical_buffer_ranges)->Unit(benchmark::kMicrosecond)->Arg(static_cast<std::int64_t>(k_buffer_steps));
BENCHMARK(mipmap_chain)->Unit(benchmark::kMicrosecond)->Arg(static_cast<std::int64_t>(k_mip_count));
BENCHMARK(individual_image_layers)->Unit(benchmark::kMicrosecond)->Arg(static_cast<std::int64_t>(k_layer_count));
BENCHMARK(multipass_uses_batches)->Unit(benchmark::kMicrosecond)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024);

}// namespace
