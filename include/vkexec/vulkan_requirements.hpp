#ifndef VKEXEC_VULKAN_REQUIREMENTS_HPP
#define VKEXEC_VULKAN_REQUIREMENTS_HPP

//! \file
//! User and library Vulkan instance/device requirements for context creation.

#include <VkBootstrap.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace vkexec {

/**
 * Type-erased Vulkan feature struct for `PhysicalDeviceSelector` / `PhysicalDevice`.
 *
 * Construct from any `VkPhysicalDevice*Features*` aggregate with an `sType` field.
 * `require` adds required extension features; `enable_if_present` enables them
 * when the device supports them.
 */
class extension_feature
{
public:
  template<typename Feature>
  // cppcheck-suppress noExplicitConstructor
  // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
  extension_feature(Feature feature)
    : require_([feature](
                 vkb::PhysicalDeviceSelector &selector) -> void { selector.add_required_extension_features(feature); }),
      enable_if_present_(
        [feature](vkb::PhysicalDevice &device) -> bool { return device.enable_extension_features_if_present(feature); })
  {}

  //! Adds this feature as required on `selector`.
  auto require(vkb::PhysicalDeviceSelector &selector) const -> void { require_(selector); }

  //! Enables this feature on `device` when present; returns whether it was enabled.
  [[nodiscard]] auto enable_if_present(vkb::PhysicalDevice &device) const -> bool { return enable_if_present_(device); }

private:
  std::function<void(vkb::PhysicalDeviceSelector &)> require_;
  std::function<bool(vkb::PhysicalDevice &)> enable_if_present_;
};

/**
 * User-requested Vulkan instance/device configuration for `context` / `window` creation.
 *
 * Library baselines from `vulkan_library` are merged in (never removed) when the
 * context is built. Raise `api_version_*` or append extensions/features as needed.
 *
 * @see context::create, scheduler_options, vulkan_library
 */
struct vulkan_requirements
{
  //! Required Vulkan API version. Raised to the library minimum if lower.
  std::uint32_t api_version_major{ 1 };
  std::uint32_t api_version_minor{ 0 };

  //! Extra instance extensions (in addition to library baselines / surface extras).
  std::vector<char const *> instance_extensions;

  //! Extra device extensions that must be present (device select fails otherwise).
  std::vector<char const *> device_extensions;

  //! Device extensions enabled when available (ignored when missing).
  std::vector<char const *> optional_device_extensions;

  //! Extra required `VkPhysicalDeviceFeatures` bits.
  VkPhysicalDeviceFeatures features{};

  //! Feature structs that must be supported (`add_required_extension_features`).
  std::vector<extension_feature> required_extension_features;

  //! Feature structs enabled when present (`enable_extension_features_if_present`).
  std::vector<extension_feature> optional_extension_features;

  /**
   * Appends a required extension feature struct.
   *
   * @param feature `VkPhysicalDevice*Features*` aggregate with `sType`.
   * @return `*this` for chaining.
   */
  template<typename Feature> auto require_extension_feature(Feature feature) -> vulkan_requirements &
  {
    required_extension_features.emplace_back(std::move(feature));
    return *this;
  }

  /**
   * Appends an optional extension feature struct.
   *
   * @param feature `VkPhysicalDevice*Features*` aggregate with `sType`.
   * @return `*this` for chaining.
   */
  template<typename Feature> auto enable_extension_feature_if_present(Feature feature) -> vulkan_requirements &
  {
    optional_extension_features.emplace_back(std::move(feature));
    return *this;
  }
};

/**
 * Library floors and extension lists always applied when vkexec creates Vulkan objects.
 *
 * Callers may request more via `vulkan_requirements`, but these baselines are not removed.
 */
namespace vulkan_library {

  constexpr std::uint32_t k_min_api_version_major = 1;
  constexpr std::uint32_t k_min_api_version_minor = 0;

  //! Instance extensions always requested for compute-only contexts.
  [[nodiscard]] auto required_instance_extensions() noexcept -> std::span<char const *const>;

  //! Instance extensions for `window::headless()` (`VK_EXT_headless_surface`).
  [[nodiscard]] auto required_headless_surface_instance_extensions() noexcept -> std::span<char const *const>;

  //! Device extensions always requested for compute-only contexts.
  [[nodiscard]] auto required_device_extensions() noexcept -> std::span<char const *const>;

  //! Device extensions required when presentation / swapchain is enabled.
  [[nodiscard]] auto required_presentation_device_extensions() noexcept -> std::span<char const *const>;

  //! Core features vkexec itself requires.
  [[nodiscard]] auto required_features() noexcept -> VkPhysicalDeviceFeatures;

}// namespace vulkan_library

}// namespace vkexec

#endif// VKEXEC_VULKAN_REQUIREMENTS_HPP
