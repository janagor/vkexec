// NOLINTBEGIN(readability-function-cognitive-complexity)

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier_params.hpp>
#include <vkexec/config.hpp>
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
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace {

constexpr VkDeviceSize k_begin = 0;
constexpr VkDeviceSize k_quarter = 128;
constexpr VkDeviceSize k_middle = 256;
constexpr VkDeviceSize k_three_quarters = 384;
constexpr VkDeviceSize k_end = 512;
constexpr std::size_t k_first_step = 0;
constexpr std::size_t k_second_step = 1;
constexpr std::size_t k_third_step = 2;
constexpr std::size_t k_fourth_step = 3;
constexpr std::size_t k_step_count = 4;
constexpr std::size_t k_one_barrier = 1;
constexpr std::size_t k_two_barriers = 2;
constexpr std::uint32_t k_family = 0;
constexpr std::uint32_t k_other_family = 1;

[[nodiscard]] auto fake_buffer(void *storage) -> VkBuffer { return static_cast<VkBuffer>(storage); }

[[nodiscard]] auto has_edge(vkexec::detail::resource_sync_plan const &plan, std::size_t source, std::size_t destination)
  -> bool
{ return std::ranges::find(plan.dependencies, std::pair{ source, destination }) != plan.dependencies.end(); }

auto require_range(vkexec::buffer_barrier_params const &barrier, VkDeviceSize offset, VkDeviceSize size) -> void
{
  REQUIRE(barrier.offset == offset);
  REQUIRE(barrier.size == size);
}

}// namespace

TEST_CASE("disjoint buffer uses remain independent including within one pass", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  for (auto const access : { vkexec::resource_access::read, vkexec::resource_access::write }) {
    vkexec::detail::resource_state_tracker tracker{ k_step_count };
    vkexec::buffer_use use{ .buffer = buffer,
      .offset = k_begin,
      .size = k_quarter,
      .usage = vkexec::buffer_usage::storage_compute,
      .access = access };
    REQUIRE(tracker.use(use, k_first_step, k_family));
    use.offset = k_middle;
    REQUIRE(tracker.use(use, k_first_step, k_family));
    use.offset = k_three_quarters;
    REQUIRE(tracker.use(use, k_second_step, k_other_family));
    auto const plan = std::move(tracker).finish();
    REQUIRE(plan.dependencies.empty());
    for (auto const &step : plan.steps) {
      REQUIRE(step.buffers_before.empty());
      REQUIRE(step.buffers_after.empty());
    }
  }
}

TEST_CASE("partial buffer read synchronizes only previously written bytes", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(tracker.use(
    vkexec::write(buffer, k_begin, k_middle, vkexec::buffer_usage::transfer_destination), k_first_step, k_family));
  REQUIRE(tracker.use(
    vkexec::read(buffer, k_quarter, k_middle, vkexec::buffer_usage::uniform_compute), k_second_step, k_family));
  REQUIRE(tracker.use(
    vkexec::read(buffer, k_begin, k_quarter, vkexec::buffer_usage::transfer_source), k_third_step, k_family));
  REQUIRE(tracker.use(
    vkexec::write(buffer, k_middle, k_quarter, vkexec::buffer_usage::storage_compute), k_fourth_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(has_edge(plan, k_first_step, k_second_step));
  REQUIRE(has_edge(plan, k_first_step, k_third_step));
  REQUIRE_FALSE(has_edge(plan, k_second_step, k_third_step));
  REQUIRE(has_edge(plan, k_second_step, k_fourth_step));
  REQUIRE_FALSE(has_edge(plan, k_first_step, k_fourth_step));
  REQUIRE(plan.steps.at(k_second_step).buffers_before.size() == k_one_barrier);
  auto const &barrier = plan.steps.at(k_second_step).buffers_before.front();
  require_range(barrier, k_quarter, k_quarter);
  REQUIRE(barrier.src_access == VK_ACCESS_2_TRANSFER_WRITE_BIT);
  REQUIRE(barrier.dst_access == VK_ACCESS_2_UNIFORM_READ_BIT);
  require_range(plan.steps.at(k_third_step).buffers_before.front(), k_begin, k_quarter);
  require_range(plan.steps.at(k_fourth_step).buffers_before.front(), k_middle, k_quarter);
}

TEST_CASE("a contained write preserves both outer buffer frontiers", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(
    tracker.use(vkexec::write(buffer, k_begin, k_end, vkexec::buffer_usage::storage_compute), k_first_step, k_family));
  REQUIRE(tracker.use(
    vkexec::write(buffer, k_quarter, k_middle, vkexec::buffer_usage::transfer_destination), k_second_step, k_family));
  REQUIRE(tracker.use(
    vkexec::read(buffer, k_begin, k_quarter, vkexec::buffer_usage::transfer_source), k_third_step, k_family));
  REQUIRE(tracker.use(
    vkexec::read(buffer, k_three_quarters, k_quarter, vkexec::buffer_usage::transfer_source), k_fourth_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(has_edge(plan, k_first_step, k_third_step));
  REQUIRE(has_edge(plan, k_first_step, k_fourth_step));
  REQUIRE_FALSE(has_edge(plan, k_second_step, k_third_step));
  REQUIRE_FALSE(has_edge(plan, k_second_step, k_fourth_step));
  require_range(plan.steps.at(k_second_step).buffers_before.front(), k_quarter, k_middle);
}

TEST_CASE("a spanning write joins every overlapping buffer reader", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(tracker.use(
    vkexec::read(buffer, k_begin, k_middle, vkexec::buffer_usage::uniform_compute), k_first_step, k_family));
  REQUIRE(tracker.use(
    vkexec::read(buffer, k_quarter, k_middle, vkexec::buffer_usage::transfer_source), k_second_step, k_family));
  REQUIRE(tracker.use(
    vkexec::read(buffer, k_middle, k_middle, vkexec::buffer_usage::uniform_fragment), k_third_step, k_family));
  REQUIRE(
    tracker.use(vkexec::write(buffer, k_begin, k_end, vkexec::buffer_usage::storage_compute), k_fourth_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(has_edge(plan, k_first_step, k_fourth_step));
  REQUIRE(has_edge(plan, k_second_step, k_fourth_step));
  REQUIRE(has_edge(plan, k_third_step, k_fourth_step));
  REQUIRE_FALSE(has_edge(plan, k_first_step, k_second_step));
  REQUIRE_FALSE(has_edge(plan, k_second_step, k_third_step));
  auto const &barriers = plan.steps.at(k_fourth_step).buffers_before;
  REQUIRE(barriers.size() == k_step_count);
  REQUIRE(barriers.at(k_second_step).src_access == (VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT));
  REQUIRE(barriers.at(k_third_step).src_access == (VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT));
}

TEST_CASE("partial ownership transfer leaves disjoint buffer ownership intact", "[vkexec][pass][ranges]")
{
  char storage{};
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  auto *const buffer = fake_buffer(&storage);
  REQUIRE(tracker.register_extent(vkexec::buffer_extent{ .buffer = buffer, .size = k_end }));
  REQUIRE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::storage_compute), k_first_step, k_family));
  REQUIRE(tracker.use(
    vkexec::read(buffer, k_quarter, k_middle, vkexec::buffer_usage::uniform_compute), k_second_step, k_other_family));
  REQUIRE(tracker.use(
    vkexec::read(buffer, k_begin, k_quarter, vkexec::buffer_usage::uniform_compute), k_third_step, k_family));
  REQUIRE(tracker.use(vkexec::read(buffer, k_three_quarters, VK_WHOLE_SIZE, vkexec::buffer_usage::uniform_compute),
    k_fourth_step,
    k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_first_step).buffers_after.size() == k_one_barrier);
  auto const &release = plan.steps.at(k_first_step).buffers_after.front();
  auto const &acquire = plan.steps.at(k_second_step).buffers_before.front();
  require_range(release, k_quarter, k_middle);
  require_range(acquire, release.offset, release.size);
  REQUIRE(release.src_queue_family == k_family);
  REQUIRE(release.dst_queue_family == k_other_family);
  REQUIRE(acquire.src_queue_family == release.src_queue_family);
  REQUIRE(acquire.dst_queue_family == release.dst_queue_family);
  REQUIRE(release.dst_stage == VK_PIPELINE_STAGE_2_NONE);
  REQUIRE(acquire.src_stage == VK_PIPELINE_STAGE_2_NONE);
  REQUIRE(plan.steps.at(k_third_step).buffers_before.front().src_queue_family == VK_QUEUE_FAMILY_IGNORED);
  require_range(plan.steps.at(k_fourth_step).buffers_before.front(), k_three_quarters, k_quarter);
  REQUIRE_FALSE(has_edge(plan, k_second_step, k_third_step));
  REQUIRE_FALSE(has_edge(plan, k_second_step, k_fourth_step));
}

TEST_CASE("partial ownership release joins and includes all relevant reader scopes", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(
    tracker.use(vkexec::read(buffer, k_begin, k_end, vkexec::buffer_usage::uniform_compute), k_first_step, k_family));
  REQUIRE(
    tracker.use(vkexec::read(buffer, k_begin, k_end, vkexec::buffer_usage::transfer_source), k_second_step, k_family));
  REQUIRE(tracker.use(
    vkexec::write(buffer, k_quarter, k_middle, vkexec::buffer_usage::storage_compute), k_third_step, k_other_family));
  REQUIRE(tracker.use(
    vkexec::write(buffer, k_begin, k_quarter, vkexec::buffer_usage::storage_compute), k_fourth_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(has_edge(plan, k_first_step, k_second_step));
  REQUIRE(has_edge(plan, k_first_step, k_fourth_step));
  REQUIRE(has_edge(plan, k_second_step, k_fourth_step));
  REQUIRE_FALSE(has_edge(plan, k_third_step, k_fourth_step));
  auto const &release = plan.steps.at(k_second_step).buffers_after.front();
  require_range(release, k_quarter, k_middle);
  REQUIRE(release.src_stage == (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT));
  REQUIRE(release.src_access == (VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT));
}

TEST_CASE("invalid buffer declarations leave all earlier regions intact", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(tracker.use(
    vkexec::write(buffer, k_begin, k_quarter, vkexec::buffer_usage::storage_compute), k_first_step, k_family));
  REQUIRE(tracker.use(
    vkexec::write(buffer, k_middle, k_quarter, vkexec::buffer_usage::storage_compute), k_second_step, k_family));
  // The first intersection would succeed; the later intersection conflicts in this step.
  REQUIRE_FALSE(
    tracker.use(vkexec::read(buffer, k_begin, k_end, vkexec::buffer_usage::uniform_compute), k_second_step, k_family));
  REQUIRE_FALSE(
    tracker.use(vkexec::read(buffer, k_begin, 0, vkexec::buffer_usage::uniform_compute), k_third_step, k_family));
  REQUIRE_FALSE(tracker.use(
    vkexec::read(
      buffer, std::numeric_limits<VkDeviceSize>::max() - k_quarter, k_middle, vkexec::buffer_usage::uniform_compute),
    k_third_step,
    k_family));
  REQUIRE(
    tracker.use(vkexec::read(buffer, k_begin, k_end, vkexec::buffer_usage::uniform_compute), k_third_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_second_step).buffers_before.empty());
  REQUIRE_FALSE(has_edge(plan, k_first_step, k_second_step));
  REQUIRE(plan.steps.at(k_third_step).buffers_before.size() == k_two_barriers);
  REQUIRE(has_edge(plan, k_first_step, k_third_step));
  REQUIRE(has_edge(plan, k_second_step, k_third_step));
}

TEST_CASE("compatible same-pass buffer reads retain every usage scope", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(tracker.use(
    vkexec::read(buffer, k_begin, k_middle, vkexec::buffer_usage::uniform_compute), k_first_step, k_family));
  REQUIRE(tracker.use(
    vkexec::read(buffer, k_quarter, k_middle, vkexec::buffer_usage::transfer_source), k_first_step, k_family));
  REQUIRE(tracker.use(
    vkexec::write(buffer, k_quarter, k_quarter, vkexec::buffer_usage::storage_compute), k_second_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(has_edge(plan, k_first_step, k_second_step));
  auto const &barrier = plan.steps.at(k_second_step).buffers_before.front();
  require_range(barrier, k_quarter, k_quarter);
  REQUIRE(barrier.src_access == (VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT));
}

TEST_CASE("identical adjacent buffer states merge before subsequent use", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(tracker.use(
    vkexec::write(buffer, k_begin, k_quarter, vkexec::buffer_usage::storage_compute), k_first_step, k_family));
  REQUIRE(tracker.use(
    vkexec::write(buffer, k_quarter, k_quarter, vkexec::buffer_usage::storage_compute), k_first_step, k_family));
  REQUIRE(tracker.use(
    vkexec::read(buffer, k_begin, k_middle, vkexec::buffer_usage::uniform_compute), k_second_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_second_step).buffers_before.size() == k_one_barrier);
  require_range(plan.steps.at(k_second_step).buffers_before.front(), k_begin, k_middle);
}

TEST_CASE("buffer interval plans match a deterministic per-byte reference", "[vkexec][pass][ranges]")
{
  constexpr std::size_t k_address_space = 64;
  constexpr std::size_t k_operation_count = 192;
  constexpr std::uint32_t k_seed = 0xB0FFEEU;
  constexpr std::size_t k_access_kind_count = 3;
  constexpr std::array<vkexec::resource_access, k_access_kind_count> k_accesses{
    vkexec::resource_access::read, vkexec::resource_access::write, vkexec::resource_access::read_write
  };
  struct byte_state
  {
    std::optional<std::size_t> writer;
    std::vector<std::size_t> readers;
  };
  struct expected_barrier
  {
    bool present{};
    VkPipelineStageFlags2 source_stage{};
    VkAccessFlags2 source_access{};
  };
  std::array<byte_state, k_address_space> bytes{};
  std::array<std::array<expected_barrier, k_address_space>, k_operation_count> expected{};
  std::array<vkexec::detail::usage_scope, k_operation_count> scopes{};
  std::set<std::pair<std::size_t, std::size_t>> edges;
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed) -- deterministic replay
  std::mt19937 random{ k_seed };
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_operation_count };
  for (std::size_t step{}; step < k_operation_count; ++step) {
    auto const begin = random() % k_address_space;
    auto const length = std::size_t{ 1 } + (random() % (k_address_space - begin));
    auto const access = k_accesses.at(random() % k_accesses.size());
    auto const usage =
      (random() & 1U) == 0 ? vkexec::buffer_usage::storage_compute : vkexec::buffer_usage::storage_graphics;
    vkexec::buffer_use const use{ .buffer = buffer,
      .offset = static_cast<VkDeviceSize>(begin),
      .size = static_cast<VkDeviceSize>(length),
      .usage = usage,
      .access = access };
    scopes.at(step) = vkexec::detail::buffer_scope(use);
    REQUIRE(tracker.use(use, step, k_family));
    for (auto address = begin; address < begin + length; ++address) {
      auto &cell = bytes.at(address);
      auto &barrier = expected.at(step).at(address);
      auto const depend = [&](std::size_t predecessor) -> void {
        edges.emplace(predecessor, step);
        barrier.present = true;
        barrier.source_stage |= scopes.at(predecessor).stage;
        barrier.source_access |= scopes.at(predecessor).access;
      };
      if (access == vkexec::resource_access::read) {
        if (cell.writer) { depend(*cell.writer); }
        cell.readers.push_back(step);
      } else {
        if (!cell.readers.empty()) {
          for (auto const reader : cell.readers) { depend(reader); }
        } else if (cell.writer) {
          depend(*cell.writer);
        }
        cell.writer = step;
        cell.readers.clear();
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
    std::array<expected_barrier, k_address_space> actual{};
    REQUIRE(plan.steps.at(step).buffers_after.empty());
    for (auto const &barrier : plan.steps.at(step).buffers_before) {
      REQUIRE(barrier.buffer == buffer);
      REQUIRE(barrier.size != 0);
      REQUIRE(barrier.offset + barrier.size <= k_address_space);
      REQUIRE(barrier.src_queue_family == VK_QUEUE_FAMILY_IGNORED);
      REQUIRE(barrier.dst_queue_family == VK_QUEUE_FAMILY_IGNORED);
      REQUIRE(barrier.dst_stage == scopes.at(step).stage);
      REQUIRE(barrier.dst_access == scopes.at(step).access);
      for (auto address = barrier.offset; address < barrier.offset + barrier.size; ++address) {
        auto &cell = actual.at(static_cast<std::size_t>(address));
        REQUIRE_FALSE(cell.present);
        cell.present = true;
        cell.source_stage = barrier.src_stage;
        cell.source_access = barrier.src_access;
      }
    }
    for (std::size_t address{}; address < k_address_space; ++address) {
      INFO("byte " << address);
      REQUIRE(actual.at(address).present == expected.at(step).at(address).present);
      REQUIRE(actual.at(address).source_stage == expected.at(step).at(address).source_stage);
      REQUIRE(actual.at(address).source_access == expected.at(step).at(address).source_access);
    }
  }
}

TEST_CASE("registered buffer extent eliminates empty whole-size tails", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(tracker.use(vkexec::uses(vkexec::buffer_extent{ .buffer = buffer, .size = k_end },
                        vkexec::write(buffer, vkexec::buffer_usage::storage_compute)),
    k_first_step,
    k_family));
  REQUIRE(
    tracker.use(vkexec::read(buffer, k_begin, k_end, vkexec::buffer_usage::uniform_compute), k_second_step, k_family));
  REQUIRE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::storage_compute), k_third_step, k_family));
  REQUIRE(tracker.region_counts().buffers == k_one_barrier);
  REQUIRE(tracker.register_extent(vkexec::buffer_extent{ .buffer = buffer, .size = k_end }));
  REQUIRE_FALSE(tracker.register_extent(vkexec::buffer_extent{ .buffer = buffer, .size = k_middle }));
  REQUIRE_FALSE(tracker.use(
    vkexec::read(buffer, k_end, VK_WHOLE_SIZE, vkexec::buffer_usage::uniform_compute), k_fourth_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_third_step).buffers_before.size() == k_one_barrier);
  require_range(plan.steps.at(k_third_step).buffers_before.front(), k_begin, k_end);
  REQUIRE(plan.steps.at(k_fourth_step).buffers_before.empty());
}

TEST_CASE("buffer extent registration rejects late registration and out of bounds uses", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  SECTION("registration must precede accesses")
  {
    vkexec::detail::resource_state_tracker tracker{ k_step_count };
    REQUIRE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::storage_compute), k_first_step, k_family));
    REQUIRE_FALSE(tracker.register_extent(vkexec::buffer_extent{ .buffer = buffer, .size = k_end }));
  }
  SECTION("registered bounds validate finite and whole ranges")
  {
    vkexec::detail::resource_state_tracker tracker{ k_step_count };
    REQUIRE(tracker.register_extent(vkexec::buffer_extent{ .buffer = buffer, .size = k_middle }));
    REQUIRE_FALSE(tracker.use(
      vkexec::write(buffer, k_quarter, k_middle, vkexec::buffer_usage::storage_compute), k_first_step, k_family));
    REQUIRE(tracker.region_counts().buffers == 0);
    REQUIRE(tracker.use(
      vkexec::write(buffer, k_quarter, VK_WHOLE_SIZE, vkexec::buffer_usage::storage_compute), k_first_step, k_family));
  }
}

TEST_CASE("concurrent buffers synchronize data without queue ownership transfers", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(tracker.register_extent(
    vkexec::buffer_extent{ .buffer = buffer, .size = k_end, .sharing_mode = VK_SHARING_MODE_CONCURRENT }));
  REQUIRE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::storage_compute), k_first_step, k_family));
  REQUIRE(tracker.use(
    vkexec::read(buffer, k_quarter, k_middle, vkexec::buffer_usage::uniform_compute), k_second_step, k_other_family));
  REQUIRE(tracker.use(
    vkexec::read(buffer, k_quarter, k_middle, vkexec::buffer_usage::uniform_fragment), k_third_step, k_family));
  REQUIRE(tracker.use(
    vkexec::write(buffer, k_quarter, k_middle, vkexec::buffer_usage::storage_compute), k_fourth_step, k_other_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(has_edge(plan, k_first_step, k_second_step));
  REQUIRE(has_edge(plan, k_first_step, k_third_step));
  REQUIRE_FALSE(has_edge(plan, k_second_step, k_third_step));
  REQUIRE(has_edge(plan, k_second_step, k_fourth_step));
  REQUIRE(has_edge(plan, k_third_step, k_fourth_step));
  for (auto const &step : plan.steps) {
    REQUIRE(step.buffers_after.empty());
    for (auto const &barrier : step.buffers_before) {
      REQUIRE(barrier.src_queue_family == VK_QUEUE_FAMILY_IGNORED);
      REQUIRE(barrier.dst_queue_family == VK_QUEUE_FAMILY_IGNORED);
    }
  }
}

TEST_CASE("buffer splits preserve queue-specific source scopes", "[vkexec][pass][ranges]")
{
  char storage{};
  char first_queue_storage{};
  char second_queue_storage{};
  auto *const buffer = fake_buffer(&storage);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  auto *const first_queue = reinterpret_cast<VkQueue>(&first_queue_storage);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  auto *const second_queue = reinterpret_cast<VkQueue>(&second_queue_storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(tracker.use(
    vkexec::write(buffer, k_begin, k_end, vkexec::buffer_usage::storage_compute), k_first_step, k_family, first_queue));
  REQUIRE(tracker.use(vkexec::read(buffer, k_quarter, k_middle, vkexec::buffer_usage::uniform_compute),
    k_second_step,
    k_family,
    second_queue));
  REQUIRE(tracker.use(vkexec::read(buffer, k_begin, k_quarter, vkexec::buffer_usage::transfer_source),
    k_third_step,
    k_family,
    first_queue));
  REQUIRE(tracker.use(vkexec::write(buffer, k_quarter, k_middle, vkexec::buffer_usage::storage_compute),
    k_fourth_step,
    k_family,
    first_queue));
  auto const plan = std::move(tracker).finish();
  REQUIRE(has_edge(plan, k_first_step, k_second_step));
  REQUIRE(has_edge(plan, k_second_step, k_fourth_step));
  REQUIRE(plan.steps.at(k_second_step).buffers_before.front().src_stage == VK_PIPELINE_STAGE_2_NONE);
  REQUIRE(plan.steps.at(k_third_step).buffers_before.front().src_stage == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  REQUIRE(plan.steps.at(k_fourth_step).buffers_before.front().src_stage == VK_PIPELINE_STAGE_2_NONE);
}

TEST_CASE("unregistered whole buffer uses reject finite splits of open tails", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::storage_compute), k_first_step, k_family));
  REQUIRE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::storage_compute), k_second_step, k_family));
  REQUIRE_FALSE(
    tracker.use(vkexec::write(buffer, k_begin, k_end, vkexec::buffer_usage::storage_compute), k_third_step, k_family));
  REQUIRE(tracker.region_counts().buffers == k_one_barrier);
  auto const plan = std::move(tracker).finish();
  REQUIRE(has_edge(plan, k_first_step, k_second_step));
  REQUIRE(plan.steps.at(k_third_step).buffers_before.empty());
  REQUIRE_FALSE(has_edge(plan, k_second_step, k_third_step));
}

TEST_CASE("registered whole buffer uses clamp away empty open tails", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(tracker.register_extent(vkexec::buffer_extent{ .buffer = buffer, .size = k_end }));
  REQUIRE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::storage_compute), k_first_step, k_family));
  REQUIRE(
    tracker.use(vkexec::write(buffer, k_begin, k_end, vkexec::buffer_usage::storage_compute), k_second_step, k_family));
  REQUIRE(tracker.use(vkexec::read(buffer, vkexec::buffer_usage::uniform_compute), k_third_step, k_family));
  REQUIRE(tracker.region_counts().buffers == k_one_barrier);
  auto const plan = std::move(tracker).finish();
  REQUIRE(has_edge(plan, k_second_step, k_third_step));
  require_range(plan.steps.at(k_third_step).buffers_before.front(), k_begin, k_end);
  REQUIRE(plan.steps.at(k_third_step).buffers_before.front().size != VK_WHOLE_SIZE);
}

TEST_CASE("uses batch rolls back earlier declarations after a later conflict", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE_FALSE(
    tracker.use(vkexec::uses(vkexec::write(buffer, k_begin, k_quarter, vkexec::buffer_usage::storage_compute),
                  vkexec::write(buffer, k_begin, k_middle, vkexec::buffer_usage::storage_compute)),
      k_first_step,
      k_family));
  REQUIRE(tracker.region_counts().buffers == 0);
  REQUIRE(
    tracker.use(vkexec::read(buffer, k_begin, k_end, vkexec::buffer_usage::uniform_compute), k_first_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.dependencies.empty());
  REQUIRE(plan.steps.at(k_first_step).buffers_before.empty());
}

TEST_CASE("unregistered finite buffer uses reject later whole-size extensions", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(
    tracker.use(vkexec::write(buffer, k_begin, k_end, vkexec::buffer_usage::storage_compute), k_first_step, k_family));
  REQUIRE_FALSE(tracker.use(vkexec::read(buffer, vkexec::buffer_usage::uniform_compute), k_second_step, k_family));
  REQUIRE(tracker.region_counts().buffers == k_one_barrier);
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_second_step).buffers_before.empty());
  REQUIRE_FALSE(has_edge(plan, k_first_step, k_second_step));
}

TEST_CASE("registered finite buffer uses accept later whole-size extensions", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(tracker.register_extent(vkexec::buffer_extent{ .buffer = buffer, .size = k_end }));
  REQUIRE(
    tracker.use(vkexec::write(buffer, k_begin, k_end, vkexec::buffer_usage::storage_compute), k_first_step, k_family));
  REQUIRE(tracker.use(vkexec::read(buffer, vkexec::buffer_usage::uniform_compute), k_second_step, k_family));
  REQUIRE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::storage_compute), k_third_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(has_edge(plan, k_first_step, k_second_step));
  REQUIRE(has_edge(plan, k_second_step, k_third_step));
  require_range(plan.steps.at(k_second_step).buffers_before.front(), k_begin, k_end);
  require_range(plan.steps.at(k_third_step).buffers_before.front(), k_begin, k_end);
  REQUIRE(plan.steps.at(k_second_step).buffers_before.front().size != VK_WHOLE_SIZE);
}

TEST_CASE("uses batch rolls back extent registration after a later conflict", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE_FALSE(tracker.use(vkexec::uses(vkexec::buffer_extent{ .buffer = buffer, .size = k_end },
                              vkexec::write(buffer, k_begin, k_quarter, vkexec::buffer_usage::storage_compute),
                              vkexec::write(buffer, k_begin, k_middle, vkexec::buffer_usage::storage_compute)),
    k_first_step,
    k_family));
  REQUIRE(tracker.region_counts().buffers == 0);
  REQUIRE(
    tracker.use(vkexec::write(buffer, k_begin, k_end, vkexec::buffer_usage::storage_compute), k_first_step, k_family));
  REQUIRE_FALSE(tracker.use(vkexec::read(buffer, vkexec::buffer_usage::uniform_compute), k_second_step, k_family));
}

TEST_CASE("uses batch rolls back ownership-release barriers after a later conflict", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(
    tracker.use(vkexec::write(buffer, k_begin, k_end, vkexec::buffer_usage::storage_compute), k_first_step, k_family));
  REQUIRE_FALSE(
    tracker.use(vkexec::uses(vkexec::read(buffer, k_quarter, k_middle, vkexec::buffer_usage::uniform_compute),
                  vkexec::write(buffer, k_quarter, k_middle, vkexec::buffer_usage::storage_compute)),
      k_second_step,
      k_other_family));
  REQUIRE(tracker.region_counts().buffers == k_one_barrier);
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_first_step).buffers_after.empty());
  REQUIRE(plan.steps.at(k_second_step).buffers_before.empty());
  REQUIRE(plan.dependencies.empty());
}

#if defined(VKEXEC_ENABLE_ALLOCATION_FAULT_INJECTION) && VKEXEC_HAS_EXCEPTIONS
// VKEXEC_HAS_EXCEPTIONS comes from <vkexec/config.hpp> (compiler unwind / __cpp_exceptions).
TEST_CASE("uses batch rolls back when new buffer insertion throws", "[vkexec][pass][ranges]")
{
  char existing_storage{};
  char created_storage{};
  auto *const existing = fake_buffer(&existing_storage);
  auto *const created = fake_buffer(&created_storage);
  vkexec::detail::resource_state_tracker tracker{ k_step_count };
  REQUIRE(tracker.use(
    vkexec::write(existing, k_begin, k_end, vkexec::buffer_usage::storage_compute), k_first_step, k_family));
  vkexec::detail::resource_allocation_fault::arm(1);
  REQUIRE_THROWS_AS(tracker.use(vkexec::uses(vkexec::buffer_extent{ .buffer = created, .size = k_end },
                                  vkexec::write(created, k_begin, k_quarter, vkexec::buffer_usage::storage_compute)),
                      k_second_step,
                      k_family),
    std::bad_alloc);
  vkexec::detail::resource_allocation_fault::disarm();
  REQUIRE(tracker.region_counts().buffers == k_one_barrier);
  REQUIRE(tracker.use(
    vkexec::write(created, k_begin, k_end, vkexec::buffer_usage::storage_compute), k_second_step, k_family));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_first_step).buffers_before.empty());
  REQUIRE(plan.steps.at(k_second_step).buffers_before.empty());
  REQUIRE(plan.dependencies.empty());
}
#endif

// NOLINTEND(readability-function-cognitive-complexity)
