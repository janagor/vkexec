#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <span>

namespace vkexec::vulkan_library {

auto required_instance_extensions() noexcept -> std::span<char const *const> { return {}; }

auto required_headless_surface_instance_extensions() noexcept -> std::span<char const *const>
{
  static constexpr std::array k_exts{
    VK_KHR_SURFACE_EXTENSION_NAME,
    VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME,
  };
  return k_exts;
}

auto required_device_extensions() noexcept -> std::span<char const *const> { return {}; }

auto required_presentation_device_extensions() noexcept -> std::span<char const *const>
{
  static constexpr std::array k_exts{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
  };
  return k_exts;
}

auto required_features() noexcept -> VkPhysicalDeviceFeatures { return {}; }

}// namespace vkexec::vulkan_library
