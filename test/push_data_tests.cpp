#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/result.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_extensions/descriptor_heap/procs.hpp>
#include <vkexec_extensions/descriptor_heap/push_data.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>
#include <utility>

namespace {

struct push_payload
{
  float x;
  std::uint32_t y;
};

}// namespace

TEST_CASE("cmd_push_data records when descriptor heap is available", "[vkexec][push_data][gpu]")
{
  VkPhysicalDeviceDescriptorHeapFeaturesEXT features_heap{};
  features_heap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
  features_heap.descriptorHeap = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  requirements.require_extension_feature(features_heap);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::context({ .requirements = std::move(requirements) }));

  REQUIRE(vkexec::descriptor_heap_available(vkexec::descriptor_heap_procs_for(*ctx)));

  auto cmd_result = ctx->allocate_command_buffer();
  REQUIRE(cmd_result.has_value());
  auto *cmd = vkexec::expected_take(cmd_result);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);

  push_payload const payload{ .x = 1.5F, .y = 9U };
  REQUIRE(vkexec::cmd_push_data(*ctx, cmd, payload));

  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
  ctx->free_command_buffer(cmd);
}
