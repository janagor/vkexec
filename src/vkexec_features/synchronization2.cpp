#include <vkexec/detail/vk_bootstrap_feature.hpp>
#include <vkexec_features/synchronization2.hpp>

#include <vkexec_features/common.hpp>
#include <vkexec_features/query.hpp>

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

namespace vkexec::feat {

auto feature_traits<synchronization2>::available(context const &ctx) -> bool
{ return detail::physical_device_synchronization2(ctx.instance(), ctx.physical_device(), ctx.api_version()); }

auto feature_traits<synchronization2>::configure(vulkan_requirements &req) -> void
{
  if (api_at_least(req, 1, 3)) {
    VkPhysicalDeviceSynchronization2Features features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
    features.synchronization2 = VK_TRUE;
    ::vkexec::detail::require_extension_feature(req, features);
    return;
  }

  if (!api_at_least(req, 1, 1)) {
    req.instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  }
  req.device_extensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
  VkPhysicalDeviceSynchronization2FeaturesKHR features{};
  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;
  features.synchronization2 = VK_TRUE;
  ::vkexec::detail::require_extension_feature(req, features);
}

}// namespace vkexec::feat
