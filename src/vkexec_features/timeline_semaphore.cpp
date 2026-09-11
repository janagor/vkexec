#include <vkexec_features/timeline_semaphore.hpp>

#include <vkexec_features/common.hpp>
#include <vkexec_features/query.hpp>

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

namespace vkexec::feat {

namespace {

  constexpr promotion k_promotion{
    .core_major = 1,
    .core_minor = 2,
    .khr_extension = VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
  };

}// namespace

auto feature_traits<timeline_semaphore>::available(context const &ctx) -> bool
{ return detail::physical_device_timeline_semaphore(ctx.physical_device(), ctx.api_version()); }

auto feature_traits<timeline_semaphore>::configure(vulkan_requirements &req) -> void
{
  if (api_at_least(req, k_promotion.core_major, k_promotion.core_minor)) {
    VkPhysicalDeviceVulkan12Features features_12{};
    features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features_12.timelineSemaphore = VK_TRUE;
    req.require_extension_feature(features_12);
    return;
  }

  req.device_extensions.push_back(k_promotion.khr_extension);
  VkPhysicalDeviceTimelineSemaphoreFeaturesKHR features_khr{};
  features_khr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR;
  features_khr.timelineSemaphore = VK_TRUE;
  req.require_extension_feature(features_khr);
}

}// namespace vkexec::feat
