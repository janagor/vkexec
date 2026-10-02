#ifndef VKEXEC_DETAIL_VK_BOOTSTRAP_FEATURE_HPP
#define VKEXEC_DETAIL_VK_BOOTSTRAP_FEATURE_HPP

#include <vkexec/vulkan_requirements.hpp>

#include <VkBootstrap.h>

#include <memory>
#include <utility>

namespace vkexec::detail {

struct extension_feature_access
{
  template<typename Feature>
  static auto make(Feature feature, capability_id capability = capability_id::none) -> extension_feature
  {
    auto const structure_type = feature.sType;
    auto state = std::make_shared<Feature>(std::move(feature));
    return extension_feature{ std::move(state),
      [](void const *value, void *selector) -> void {
        static_cast<vkb::PhysicalDeviceSelector *>(selector)->add_required_extension_features(
          *static_cast<Feature const *>(value));
      },
      [](void const *value, void *device) -> bool {
        return static_cast<vkb::PhysicalDevice *>(device)->enable_extension_features_if_present(
          *static_cast<Feature const *>(value));
      },
      structure_type,
      capability };
  }

  static auto s_type(extension_feature const &feature) noexcept -> VkStructureType { return feature.s_type_; }
  static auto capability(extension_feature const &feature) noexcept -> capability_id { return feature.capability_; }

  static auto require(extension_feature const &feature, vkb::PhysicalDeviceSelector &selector) -> void
  { feature.require_(feature.state_.get(), &selector); }

  static auto enable_if_present(extension_feature const &feature, vkb::PhysicalDevice &device) -> bool
  { return feature.enable_(feature.state_.get(), &device); }
};

template<typename Feature>
auto require_extension_feature(vulkan_requirements &requirements,
  Feature feature,
  capability_id capability = capability_id::none) -> void
{ requirements.required_extension_features.push_back(extension_feature_access::make(std::move(feature), capability)); }

template<typename Feature>
auto enable_extension_feature_if_present(vulkan_requirements &requirements,
  Feature feature,
  capability_id capability = capability_id::none) -> void
{ requirements.optional_extension_features.push_back(extension_feature_access::make(std::move(feature), capability)); }

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_VK_BOOTSTRAP_FEATURE_HPP
