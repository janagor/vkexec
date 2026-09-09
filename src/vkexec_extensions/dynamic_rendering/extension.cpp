#include <vkexec_extensions/dynamic_rendering/extension.hpp>

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

namespace vkexec::ext {

auto extension_traits<dynamic_rendering>::available(context const &ctx) -> bool
{ return VK_API_VERSION_MINOR(ctx.api_version()) >= 3; }

auto extension_traits<dynamic_rendering>::configure(vulkan_requirements &req) -> void
{
  VkPhysicalDeviceVulkan13Features features_13{};
  features_13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  features_13.dynamicRendering = VK_TRUE;
  req.require_extension_feature(features_13);
}

}// namespace vkexec::ext
