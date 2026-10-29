#include "graph_submission_reference.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <type_traits>

namespace {

template<class Handle> [[nodiscard]] auto fake_handle(std::uintptr_t *storage) -> Handle
{
  if constexpr (std::is_pointer_v<Handle>) {
    return reinterpret_cast<Handle>(storage);
  } else {
    return static_cast<Handle>(reinterpret_cast<std::uintptr_t>(storage));
  }
}

auto require_entries(std::span<vkexec::semaphore_submit const> actual,
  std::span<vkexec::semaphore_submit const> expected) -> void
{
  REQUIRE(actual.size() == expected.size());
  for (std::size_t index{}; index < actual.size(); ++index) {
    REQUIRE(actual[index].semaphore == expected[index].semaphore);
    REQUIRE(actual[index].value == expected[index].value);
    REQUIRE(actual[index].stage == expected[index].stage);
  }
}

}// namespace

TEST_CASE("prepared graph synchronization matches edge-scanning reference", "[vkexec][pass][sync]")
{
  constexpr std::array k_sizes{ std::size_t{}, std::size_t{ 1 }, std::size_t{ 8 }, std::size_t{ 32 } };
  constexpr std::array k_shapes{ vkexec::test::submission_shape::linear,
    vkexec::test::submission_shape::independent,
    vkexec::test::submission_shape::fork_join,
    vkexec::test::submission_shape::dense,
    vkexec::test::submission_shape::fan_in,
    vkexec::test::submission_shape::mixed_queue };
  constexpr std::array k_modes{ false, true };
  for (auto const count : k_sizes) {
    for (auto const shape : k_shapes) {
      auto plan = vkexec::test::submission_plan(count, shape);
      if (shape == vkexec::test::submission_shape::dense) {
        for (auto &batch : plan.batches) { std::ranges::reverse(batch.predecessors); }
      }
      for (bool const timeline : k_modes) {
        auto const semaphore_count = timeline ? count : vkexec::detail::graph_dependency_count(plan);
        // Real, distinct host addresses make opaque test handles without Vulkan calls.
        std::vector<VkSemaphore> handles(semaphore_count);
        std::vector<std::uintptr_t> storage(semaphore_count);
        for (std::size_t index{}; index < semaphore_count; ++index) {
          handles.at(index) = fake_handle<VkSemaphore>(&storage.at(index));
        }
        for (bool const presentation : k_modes) {
          std::uintptr_t acquire_storage{};
          std::uintptr_t finished_storage{};
          vkexec::semaphore_submit const acquire{ .semaphore = fake_handle<VkSemaphore>(&acquire_storage),
            .stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT };
          vkexec::semaphore_submit const finished{ .semaphore = fake_handle<VkSemaphore>(&finished_storage) };
          auto const *wait = presentation ? &acquire : nullptr;
          auto const *signal = presentation ? &finished : nullptr;
          auto const reference = vkexec::test::reference_batch_sync(plan, handles, timeline, wait, signal);
          auto const prepared = vkexec::detail::prepare_batch_sync(plan, handles, timeline, wait, signal);
          CAPTURE(count, timeline, presentation);
          REQUIRE(prepared.wait_offsets == reference.wait_offsets);
          REQUIRE(prepared.signal_offsets == reference.signal_offsets);
          for (std::size_t batch{}; batch < count; ++batch) {
            require_entries(prepared.waits_for(batch), reference.waits_for(batch));
            require_entries(prepared.signals_for(batch), reference.signals_for(batch));
          }
        }
      }
    }
  }
}
