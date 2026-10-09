#include <vkexec/detail/capability_id.hpp>
#include <vkexec/detail/vk_bootstrap_feature.hpp>
#include <vkexec_features/common.hpp>
#include <vkexec_features/shader_demote_to_helper_invocation.hpp>

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

namespace vkexec::feat {

auto feature_traits<shader_demote_to_helper_invocation>::available(context const &ctx) -> bool
{ return ctx.capabilities().shader_demote_to_helper_invocation; }

auto feature_traits<shader_demote_to_helper_invocation>::configure(vulkan_requirements &req) -> void
{
  if (!api_at_least(req, 1, 3)) {
    req.device_extensions.push_back(VK_EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_EXTENSION_NAME);
  }
  VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures features{};
  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES;
  features.shaderDemoteToHelperInvocation = VK_TRUE;
  ::vkexec::detail::require_extension_feature(
    req, features, ::vkexec::detail::capability_id::shader_demote_to_helper_invocation);
}

}// namespace vkexec::feat
