#ifndef VKEXEC_EXAMPLES_VULKAN_SAMPLES_COMMON_VULKAN_REQUIREMENTS_HPP
#define VKEXEC_EXAMPLES_VULKAN_SAMPLES_COMMON_VULKAN_REQUIREMENTS_HPP

#include <vkexec/vulkan_requirements.hpp>

namespace vkexec::examples {

[[nodiscard]] inline auto vulkan_sample_requirements() -> vulkan_requirements
{
  vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 3;
  return requirements;
}

}// namespace vkexec::examples

#endif
