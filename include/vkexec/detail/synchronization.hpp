#ifndef VKEXEC_DETAIL_SYNCHRONIZATION_HPP
#define VKEXEC_DETAIL_SYNCHRONIZATION_HPP

#include <vkexec/result.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>

namespace vkexec::detail {

enum class synchronization_backend : std::uint8_t
{
  legacy,
  synchronization2_khr,
  synchronization2_core,
};

[[nodiscard]] constexpr auto select_synchronization_backend(std::uint32_t api_version,
  bool khr_extension_enabled,
  bool feature_enabled,
  bool instance_dependency_enabled) noexcept -> synchronization_backend
{
  if (!feature_enabled) { return synchronization_backend::legacy; }
  if (api_version >= VK_API_VERSION_1_3) { return synchronization_backend::synchronization2_core; }
  if (khr_extension_enabled && (api_version >= VK_API_VERSION_1_1 || instance_dependency_enabled)) {
    return synchronization_backend::synchronization2_khr;
  }
  return synchronization_backend::legacy;
}

[[nodiscard]] auto legacy_stage_mask(VkPipelineStageFlags2 value) -> result<VkPipelineStageFlags>;
[[nodiscard]] auto legacy_access_mask(VkAccessFlags2 value) -> result<VkAccessFlags>;
[[nodiscard]] auto legacy_scope_stage(VkPipelineStageFlags2 stage, VkAccessFlags2 access, bool source)
  -> result<VkPipelineStageFlags>;
[[nodiscard]] auto validate_scope(VkPipelineStageFlags2 stage, VkAccessFlags2 access) -> status;

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_SYNCHRONIZATION_HPP
