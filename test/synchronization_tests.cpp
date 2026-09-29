#include <catch2/catch_test_macros.hpp>

#include <vkexec/detail/submission.hpp>
#include <vkexec/detail/synchronization.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_features/feature.hpp>
#include <vkexec_features/synchronization2.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cstring>

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

TEST_CASE("synchronization backend requires enabled feature and extension", "[vkexec][sync]")
{
  using vkexec::detail::select_synchronization_backend;
  using vkexec::detail::synchronization_backend;
  REQUIRE(select_synchronization_backend(VK_API_VERSION_1_3, false, true, false)
          == synchronization_backend::synchronization2_core);
  REQUIRE(select_synchronization_backend(VK_API_VERSION_1_2, true, true, true)
          == synchronization_backend::synchronization2_khr);
  REQUIRE(select_synchronization_backend(VK_API_VERSION_1_2, false, true, true) == synchronization_backend::legacy);
  REQUIRE(select_synchronization_backend(VK_API_VERSION_1_2, true, false, true) == synchronization_backend::legacy);
  REQUIRE(select_synchronization_backend(VK_API_VERSION_1_0, true, true, true)
          == synchronization_backend::synchronization2_khr);
  REQUIRE(select_synchronization_backend(VK_API_VERSION_1_0, true, true, false) == synchronization_backend::legacy);
  REQUIRE(select_synchronization_backend(VK_API_VERSION_1_0, false, true, true) == synchronization_backend::legacy);
}

TEST_CASE("Vulkan 1.0 synchronization2 requirement includes instance dependency", "[vkexec][sync]")
{
  vkexec::vulkan_requirements requirements{};
  vkexec::feat::configure<vkexec::feat::synchronization2>(requirements);
  REQUIRE(std::ranges::any_of(requirements.instance_extensions, [](char const *extension) -> bool {
    return std::strcmp(extension, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == 0;
  }));
  REQUIRE(std::ranges::any_of(requirements.device_extensions, [](char const *extension) -> bool {
    return std::strcmp(extension, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0;
  }));
  REQUIRE(requirements.required_extension_features.size() == 1);
}

TEST_CASE("Vulkan 1.2 synchronization2 requires only the KHR device extension", "[vkexec][sync]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_minor = 2;
  vkexec::feat::configure<vkexec::feat::synchronization2>(requirements);
  REQUIRE(requirements.instance_extensions.empty());
  REQUIRE(std::ranges::any_of(requirements.device_extensions, [](char const *extension) -> bool {
    return std::strcmp(extension, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0;
  }));
  REQUIRE(requirements.required_extension_features.size() == 1);
}

TEST_CASE("Vulkan 1.3 synchronization2 requires no KHR extensions", "[vkexec][sync]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_minor = 3;
  vkexec::feat::configure<vkexec::feat::synchronization2>(requirements);
  REQUIRE(requirements.instance_extensions.empty());
  REQUIRE(requirements.device_extensions.empty());
  REQUIRE(requirements.required_extension_features.size() == 1);
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
  REQUIRE(binary_view.submit.pWaitDstStageMask[0] == VK_PIPELINE_STAGE_TRANSFER_BIT);
  REQUIRE(binary_view.submit.pWaitSemaphores != nullptr);

  auto timeline_enabled = vkexec::detail::lower_legacy_submit(request, true);
  REQUIRE(timeline_enabled);
  auto const view = timeline_enabled->make_submit_info(request);
  REQUIRE(view.submit.pNext == &view.timeline);
  REQUIRE(view.timeline.waitSemaphoreValueCount == 1);
  REQUIRE(view.timeline.pWaitSemaphoreValues[0] == 0);
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
  REQUIRE(lowered.commands[0].deviceMask == 1);
  REQUIRE(lowered.waits[0].stageMask == VK_PIPELINE_STAGE_2_COPY_BIT);
  REQUIRE(lowered.waits[0].value == 17);
  REQUIRE(lowered.signals[0].stageMask == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  REQUIRE(lowered.signals[0].value == 23);
  auto const submission = lowered.submit_info();
  REQUIRE(submission.commandBufferInfoCount == 1);
  REQUIRE(submission.pCommandBufferInfos == lowered.commands.data());
  REQUIRE(submission.pWaitSemaphoreInfos == lowered.waits.data());
  REQUIRE(submission.pSignalSemaphoreInfos == lowered.signals.data());
}
