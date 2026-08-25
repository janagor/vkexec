#ifndef VKEXEC_VULKAN_REQUIREMENTS_HPP
#define VKEXEC_VULKAN_REQUIREMENTS_HPP

#include <vulkan/vulkan.h>

#include <cstdint>
#include <span>
#include <vector>

namespace vkexec {

/// User-requested Vulkan instance/device configuration for `context` / `window` creation.
/// Library baselines are merged in (never removed) when the context is built.
struct vulkan_requirements
{
  /// Required Vulkan API version. Raised to the library minimum if lower.
  std::uint32_t api_version_major{ 1 };
  std::uint32_t api_version_minor{ 0 };

  /// Extra instance extensions (in addition to library baselines / surface extras).
  std::vector<char const *> instance_extensions;

  /// Extra device extensions (in addition to library baselines; presentation adds swapchain).
  std::vector<char const *> device_extensions;

  /// Extra required `VkPhysicalDeviceFeatures` bits (OR'd with library baselines).
  VkPhysicalDeviceFeatures features{};
};

/// Library floors and extension lists always applied when vkexec creates Vulkan objects.
namespace vulkan_library {

  constexpr std::uint32_t k_min_api_version_major = 1;
  constexpr std::uint32_t k_min_api_version_minor = 0;

  /// Instance extensions always requested for compute-only contexts.
  [[nodiscard]] inline auto required_instance_extensions() noexcept -> std::span<char const * const>
  { return {}; }

  /// Instance extensions for `window::headless()` (`VK_EXT_headless_surface`).
  [[nodiscard]] inline auto required_headless_surface_instance_extensions() noexcept
    -> std::span<char const * const>
  {
    static constexpr char const *exts[] = {
      VK_KHR_SURFACE_EXTENSION_NAME,
      VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME,
    };
    return exts;
  }

  /// Device extensions always requested for compute-only contexts.
  [[nodiscard]] inline auto required_device_extensions() noexcept -> std::span<char const * const> { return {}; }

  /// Device extensions required when presentation / swapchain is enabled.
  [[nodiscard]] inline auto required_presentation_device_extensions() noexcept -> std::span<char const * const>
  {
    static constexpr char const *exts[] = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };
    return exts;
  }

  /// Core features vkexec itself requires (OR'd into the user request).
  [[nodiscard]] inline auto required_features() noexcept -> VkPhysicalDeviceFeatures { return {}; }

}// namespace vulkan_library

}// namespace vkexec

#endif// VKEXEC_VULKAN_REQUIREMENTS_HPP
