#include <vkexec_features/buffer_device_address.hpp>

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
    .khr_extension = VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
  };

}// namespace

auto feature_traits<buffer_device_address>::available(context const &ctx) -> bool
{
  if (ctx.procs().get_buffer_device_address != nullptr) { return true; }
  return detail::physical_device_buffer_device_address(ctx.physical_device(), ctx.api_version());
}

auto feature_traits<buffer_device_address>::configure(vulkan_requirements &req) -> void
{
  if (api_at_least(req, k_promotion.core_major, k_promotion.core_minor)) {
    VkPhysicalDeviceVulkan12Features features_12{};
    features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features_12.bufferDeviceAddress = VK_TRUE;
    req.require_extension_feature(features_12);
    return;
  }

  req.device_extensions.push_back(k_promotion.khr_extension);
  VkPhysicalDeviceBufferDeviceAddressFeaturesKHR features_khr{};
  features_khr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR;
  features_khr.bufferDeviceAddress = VK_TRUE;
  req.require_extension_feature(features_khr);
}

}// namespace vkexec::feat
