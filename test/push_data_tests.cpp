#include <catch2/catch_test_macros.hpp>

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/push_data.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace {

struct push_payload
{
  float x;
  std::uint32_t y;
};

auto skip_if_unavailable(vkexec::error const &err) -> void
{ SKIP(std::string("Vulkan feature set unavailable: ") + std::string(err.message())); }

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

  auto ctx_result = vkexec::context::create({ .requirements = std::move(requirements) });
  if (!ctx_result) { skip_if_unavailable(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  REQUIRE(ctx.procs().cmd_push_data != nullptr);

  auto cmd_result = ctx.allocate_command_buffer();
  REQUIRE(cmd_result.has_value());
  VkCommandBuffer cmd = *cmd_result;
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);

  push_payload const payload{ .x = 1.5F, .y = 9U };
  REQUIRE(vkexec::cmd_push_data(ctx, cmd, payload));

  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
  ctx.free_command_buffer(cmd);
}
