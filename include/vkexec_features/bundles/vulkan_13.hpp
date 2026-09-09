#ifndef VKEXEC_FEATURES_BUNDLES_VULKAN_13_HPP
#define VKEXEC_FEATURES_BUNDLES_VULKAN_13_HPP

#include <vkexec_features/bundles/vulkan_12.hpp>
#include <vkexec_features/dynamic_rendering.hpp>
#include <vkexec_features/feature.hpp>

namespace vkexec::feat {

inline auto configure_vulkan_13(vulkan_requirements &req) -> void
{
  configure_vulkan_12(req);
  configure<dynamic_rendering>(req);
}

}// namespace vkexec::feat

#endif// VKEXEC_FEATURES_BUNDLES_VULKAN_13_HPP
