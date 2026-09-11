#include <vkexec_features/dynamic_rendering.hpp>

#include <vkexec_features/common.hpp>
#include <vkexec_features/query.hpp>

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

namespace vkexec::feat {

namespace {

  constexpr promotion k_promotion{
    .core_major = 1,
    .core_minor = 3,
    .khr_extension = VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
  };

}// namespace

auto feature_traits<dynamic_rendering>::available(context const &ctx) -> bool
{ return detail::physical_device_dynamic_rendering(ctx.physical_device(), ctx.api_version()); }

auto feature_traits<dynamic_rendering>::configure(vulkan_requirements &req) -> void
{
  // Prefer core 1.3 feature struct; otherwise require the KHR extension + feature.
  if (api_at_least(req, k_promotion.core_major, k_promotion.core_minor)) {
    VkPhysicalDeviceVulkan13Features features_13{};
    features_13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features_13.dynamicRendering = VK_TRUE;
    req.require_extension_feature(features_13);
    return;
  }

  req.device_extensions.push_back(k_promotion.khr_extension);
  VkPhysicalDeviceDynamicRenderingFeaturesKHR features_khr{};
  features_khr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR;
  features_khr.dynamicRendering = VK_TRUE;
  req.require_extension_feature(features_khr);
}

}// namespace vkexec::feat
