#ifndef VKEXEC_FEATURES_BUNDLES_VULKAN_12_HPP
#define VKEXEC_FEATURES_BUNDLES_VULKAN_12_HPP

#include <vkexec_features/buffer_device_address.hpp>
#include <vkexec_features/feature.hpp>
#include <vkexec_features/timeline_semaphore.hpp>

namespace vkexec::feat {

inline auto configure_vulkan_12(vulkan_requirements &req) -> void
{
  configure<timeline_semaphore>(req);
  configure<buffer_device_address>(req);
}

}// namespace vkexec::feat

#endif// VKEXEC_FEATURES_BUNDLES_VULKAN_12_HPP
