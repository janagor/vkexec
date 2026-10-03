#include <catch2/catch_test_macros.hpp>

#include <vkexec/detail/object_synchronization.hpp>
#include <vkexec/detail/submission.hpp>
#include <vkexec/detail/synchronization.hpp>
#include <vkexec/device_capabilities.hpp>
#include <vkexec/queue_submit.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <atomic>
#include <mutex>
#include <thread>

TEST_CASE("object synchronization shares state for aliased handles", "[vkexec][sync]")
{
  vkexec::detail::object_synchronization_registry<int, vkexec::detail::queue_synchronization_state> registry;
  auto first = registry.state(1);
  auto alias = registry.state(1);
  auto independent = registry.state(2);
  REQUIRE(first == alias);
  REQUIRE(first != independent);
}

TEST_CASE("object synchronization blocks one handle without blocking another", "[vkexec][sync]")
{
  vkexec::detail::object_synchronization_registry<int> registry;
  auto held = registry.state(1);
  std::unique_lock const lock(*held);
  std::atomic<bool> same_acquired{ true };
  std::atomic<bool> other_acquired{ false };
  std::thread same([&]() -> void {
    std::unique_lock const attempt(*registry.state(1), std::try_to_lock);
    same_acquired.store(attempt.owns_lock());
  });
  std::thread other([&]() -> void {
    std::unique_lock const attempt(*registry.state(2), std::try_to_lock);
    other_acquired.store(attempt.owns_lock());
  });
  same.join();
  other.join();
  REQUIRE_FALSE(same_acquired.load());
  REQUIRE(other_acquired.load());
}

TEST_CASE("synchronization2 stages lower to legacy stages", "[vkexec][sync]")
{
  using vkexec::detail::legacy_stage_mask;
  REQUIRE(legacy_stage_mask(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT).value() == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  REQUIRE(legacy_stage_mask(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT).value() == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
  REQUIRE(legacy_stage_mask(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT).value() == VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
  REQUIRE(legacy_stage_mask(VK_PIPELINE_STAGE_2_COPY_BIT).value() == VK_PIPELINE_STAGE_TRANSFER_BIT);
  REQUIRE(legacy_stage_mask(VK_PIPELINE_STAGE_2_BLIT_BIT).value() == VK_PIPELINE_STAGE_TRANSFER_BIT);
  REQUIRE(legacy_stage_mask(VK_PIPELINE_STAGE_2_RESOLVE_BIT).value() == VK_PIPELINE_STAGE_TRANSFER_BIT);
  REQUIRE(legacy_stage_mask(VK_PIPELINE_STAGE_2_CLEAR_BIT).value() == VK_PIPELINE_STAGE_TRANSFER_BIT);
  REQUIRE(legacy_stage_mask(VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT).value() == VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
  REQUIRE(
    legacy_stage_mask(VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT).value() == VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
  REQUIRE_FALSE(legacy_stage_mask(VkPipelineStageFlags2{ 0x8000000000000000ULL }));
}

TEST_CASE("synchronization2 accesses lower to legacy accesses", "[vkexec][sync]")
{
  using vkexec::detail::legacy_access_mask;
  REQUIRE(legacy_access_mask(VK_ACCESS_2_UNIFORM_READ_BIT).value() == VK_ACCESS_SHADER_READ_BIT);
  REQUIRE(legacy_access_mask(VK_ACCESS_2_SHADER_SAMPLED_READ_BIT).value() == VK_ACCESS_SHADER_READ_BIT);
  REQUIRE(legacy_access_mask(VK_ACCESS_2_SHADER_STORAGE_READ_BIT).value() == VK_ACCESS_SHADER_READ_BIT);
  REQUIRE(legacy_access_mask(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT).value() == VK_ACCESS_SHADER_WRITE_BIT);
  REQUIRE_FALSE(legacy_access_mask(VkAccessFlags2{ 0x8000000000000000ULL }));
}

TEST_CASE("NONE stages have legacy scope equivalents only with empty access", "[vkexec][sync]")
{
  using vkexec::detail::legacy_scope_stage;
  REQUIRE(
    legacy_scope_stage(VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, true).value() == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
  REQUIRE(legacy_scope_stage(VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, false).value()
          == VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
  REQUIRE_FALSE(legacy_scope_stage(VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_SHADER_READ_BIT, true));
  REQUIRE_FALSE(vkexec::detail::validate_scope(VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_SHADER_READ_BIT));
}

TEST_CASE("synchronization backend follows enabled capability", "[vkexec][sync]")
{
  using vkexec::detail::select_synchronization_backend;
  using vkexec::detail::synchronization_backend;
  vkexec::device_capabilities caps{};
  REQUIRE(select_synchronization_backend(caps) == synchronization_backend::legacy);
  caps.synchronization2 = true;
  REQUIRE(select_synchronization_backend(caps) == synchronization_backend::synchronization2_khr);
  caps.api_version = VK_API_VERSION_1_3;
  REQUIRE(select_synchronization_backend(caps) == synchronization_backend::synchronization2_core);
  caps.synchronization2 = false;
  REQUIRE(select_synchronization_backend(caps) == synchronization_backend::legacy);
}

TEST_CASE("submission stages preserve synchronization2 semantics", "[vkexec][sync]")
{
  REQUIRE(vkexec::semaphore_submit{}.stage == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
  REQUIRE(vkexec::detail::legacy_wait_stage(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT).value()
          == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  REQUIRE(vkexec::detail::legacy_wait_stage(VK_PIPELINE_STAGE_2_COPY_BIT).value() == VK_PIPELINE_STAGE_TRANSFER_BIT);
  REQUIRE_FALSE(vkexec::detail::legacy_wait_stage(VK_PIPELINE_STAGE_2_NONE));
  REQUIRE_FALSE(vkexec::detail::legacy_wait_stage(VK_PIPELINE_STAGE_2_HOST_BIT));
  REQUIRE_FALSE(vkexec::detail::legacy_wait_stage(VkPipelineStageFlags2{ 0x8000000000000000ULL }));
  REQUIRE(vkexec::detail::legacy_signal_stage(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT));
  REQUIRE_FALSE(vkexec::detail::legacy_signal_stage(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT));
}

TEST_CASE("legacy submission attaches timeline values only when supported", "[vkexec][sync]")
{
  std::array<VkCommandBuffer, 1> const commands{ VK_NULL_HANDLE };
  std::array<vkexec::semaphore_submit, 1> const waits{ vkexec::semaphore_submit{
    .value = 0, .stage = VK_PIPELINE_STAGE_2_COPY_BIT } };
  vkexec::queue_submit const request{ .command_buffers = commands, .waits = waits };

  auto binary_only = vkexec::detail::lower_legacy_submit(request, false);
  REQUIRE(binary_only);
  auto const binary_view = binary_only->make_submit_info(request);
  REQUIRE(binary_view.submit.pNext == nullptr);
  REQUIRE(binary_only->wait_stages.at(0) == VK_PIPELINE_STAGE_TRANSFER_BIT);
  REQUIRE(binary_view.submit.pWaitDstStageMask == binary_only->wait_stages.data());
  REQUIRE(binary_view.submit.pWaitSemaphores != nullptr);

  auto timeline_enabled = vkexec::detail::lower_legacy_submit(request, true);
  REQUIRE(timeline_enabled);
  auto const view = timeline_enabled->make_submit_info(request);
  REQUIRE(view.submit.pNext == &view.timeline);
  REQUIRE(view.timeline.waitSemaphoreValueCount == 1);
  REQUIRE(timeline_enabled->wait_values.at(0) == 0);
  REQUIRE(view.timeline.pWaitSemaphoreValues == timeline_enabled->wait_values.data());
  REQUIRE(view.timeline.signalSemaphoreValueCount == 0);
  REQUIRE(view.timeline.pSignalSemaphoreValues == nullptr);
}

TEST_CASE("synchronization2 submission lowering preserves stages and values", "[vkexec][sync]")
{
  std::array<VkCommandBuffer, 1> const commands{ VK_NULL_HANDLE };
  std::array<vkexec::semaphore_submit, 1> const waits{ vkexec::semaphore_submit{
    .value = 17, .stage = VK_PIPELINE_STAGE_2_COPY_BIT } };
  std::array<vkexec::semaphore_submit, 1> const signals{ vkexec::semaphore_submit{
    .value = 23, .stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT } };
  vkexec::queue_submit const request{ .command_buffers = commands, .waits = waits, .signals = signals };

  auto const lowered = vkexec::detail::lower_synchronization2_submit(request);
  REQUIRE(lowered.commands.size() == 1);
  REQUIRE(lowered.commands.at(0).deviceMask == 1);
  REQUIRE(lowered.waits.at(0).stageMask == VK_PIPELINE_STAGE_2_COPY_BIT);
  REQUIRE(lowered.waits.at(0).value == 17);
  REQUIRE(lowered.signals.at(0).stageMask == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  REQUIRE(lowered.signals.at(0).value == 23);
  auto const submission = lowered.submit_info();
  REQUIRE(submission.commandBufferInfoCount == 1);
  REQUIRE(submission.pCommandBufferInfos == lowered.commands.data());
  REQUIRE(submission.pWaitSemaphoreInfos == lowered.waits.data());
  REQUIRE(submission.pSignalSemaphoreInfos == lowered.signals.data());
}
