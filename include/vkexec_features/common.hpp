#ifndef VKEXEC_FEATURES_COMMON_HPP
#define VKEXEC_FEATURES_COMMON_HPP

#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>

namespace vkexec::feat {

struct promotion
{
  std::uint32_t core_major{ 1 };
  std::uint32_t core_minor{ 0 };
  char const *khr_extension{ nullptr };
};

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
[[nodiscard]] constexpr auto
  api_at_least(vulkan_requirements const &req, std::uint32_t major, std::uint32_t minor) noexcept -> bool
{
  if (req.api_version_major != major) { return req.api_version_major > major; }
  return req.api_version_minor >= minor;
}

[[nodiscard]] constexpr auto api_at_least(std::uint32_t api_version, std::uint32_t major, std::uint32_t minor) noexcept
  -> bool
{
  if (VK_API_VERSION_MAJOR(api_version) != major) { return VK_API_VERSION_MAJOR(api_version) > major; }
  return VK_API_VERSION_MINOR(api_version) >= minor;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

}// namespace vkexec::feat

#endif// VKEXEC_FEATURES_COMMON_HPP
