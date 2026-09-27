#ifndef VKEXEC_VULKAN_REQUIREMENTS_HPP
#define VKEXEC_VULKAN_REQUIREMENTS_HPP

//! \file
//! User and library Vulkan instance/device requirements for context creation.

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace vkexec {

namespace detail {
  struct extension_feature_access;
}

//! Opaque feature request created by vkexec's compiled feature modules.
class extension_feature
{
public:
  extension_feature(extension_feature const &) noexcept = default;
  extension_feature(extension_feature &&) noexcept = default;
  auto operator=(extension_feature const &) noexcept -> extension_feature & = default;
  auto operator=(extension_feature &&) noexcept -> extension_feature & = default;
  ~extension_feature() = default;

private:
  friend struct detail::extension_feature_access;
  extension_feature(std::shared_ptr<void const> state,
    void (*require)(void const *, void *),
    bool (*enable)(void const *, void *)) noexcept
    : state_(std::move(state)), require_(require), enable_(enable)
  {}

  std::shared_ptr<void const> state_;
  void (*require_)(void const *, void *){};
  bool (*enable_)(void const *, void *){};
};

/**
 * User-requested Vulkan instance/device configuration for `context` / `window` creation.
 *
 * Library baselines from `vulkan_library` are merged in (never removed) when the
 * context is built. Raise `api_version_*` or append extensions/features as needed.
 *
 * @see factory::make_context, scheduler_options, vulkan_library
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

  //! Feature requests that must be supported.
  std::vector<extension_feature> required_extension_features;

  //! Feature requests enabled when present.
  std::vector<extension_feature> optional_extension_features;
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

  //! Instance extensions for `factory::make_headless_presenter()` (`VK_EXT_headless_surface`).
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
