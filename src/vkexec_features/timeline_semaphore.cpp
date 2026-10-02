#include <vkexec/detail/capability_id.hpp>
#include <vkexec/detail/vk_bootstrap_feature.hpp>
#include <vkexec_features/timeline_semaphore.hpp>

#include <vkexec_features/common.hpp>

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
{ return ctx.capabilities().timeline_semaphore; }

auto feature_traits<timeline_semaphore>::configure(vulkan_requirements &req) -> void
{
  // Prefer core 1.2 feature struct; otherwise require the KHR extension + feature.
  if (api_at_least(req, k_promotion.core_major, k_promotion.core_minor)) {
    VkPhysicalDeviceVulkan12Features features_12{};
    features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features_12.timelineSemaphore = VK_TRUE;
    ::vkexec::detail::require_extension_feature(req, features_12, ::vkexec::detail::capability_id::timeline_semaphore);
    return;
  }

  req.device_extensions.push_back(k_promotion.khr_extension);
  VkPhysicalDeviceTimelineSemaphoreFeaturesKHR features_khr{};
  features_khr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR;
  features_khr.timelineSemaphore = VK_TRUE;
  ::vkexec::detail::require_extension_feature(req, features_khr, ::vkexec::detail::capability_id::timeline_semaphore);
}

}// namespace vkexec::feat
