#include <catch2/catch_test_macros.hpp>

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/push_data.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <utility>

namespace {

struct push_payload
{
  float x;
  std::uint32_t y;
};

auto skip_if_unavailable(std::exception const &error) -> void
{ SKIP(std::string("Vulkan feature set unavailable: ") + error.what()); }

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

  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(vkexec::scheduler_options{ .requirements = std::move(requirements) }); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_unavailable(error); }

  REQUIRE(ctx->procs().cmd_push_data != nullptr);

  VkCommandBuffer cmd = ctx->allocate_command_buffer();
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);

  push_payload const payload{ .x = 1.5F, .y = 9U };
  vkexec::cmd_push_data(*ctx, cmd, payload);

  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
  ctx->free_command_buffer(cmd);
}
