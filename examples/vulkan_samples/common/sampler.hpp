#ifndef VKEXEC_EXAMPLES_SAMPLER_HPP
#define VKEXEC_EXAMPLES_SAMPLER_HPP

#include "sync_wait_helpers.hpp"

#include <vulkan/vulkan_core.h>

namespace vkexec::examples {

struct sampler_owner
{
  VkDevice device{ VK_NULL_HANDLE };
  VkSampler handle{ VK_NULL_HANDLE };
  sampler_owner(VkDevice owned_device, VkSampler owned_handle) noexcept : device(owned_device), handle(owned_handle) {}
  sampler_owner(sampler_owner const &) = delete;
  auto operator=(sampler_owner const &) -> sampler_owner & = delete;
  sampler_owner(sampler_owner &&) = delete;
  auto operator=(sampler_owner &&) -> sampler_owner & = delete;
  ~sampler_owner()
  {
    if (handle != VK_NULL_HANDLE) { vkDestroySampler(device, handle, nullptr); }
  }
};

[[nodiscard]] inline auto make_sampler(VkDevice device,
  VkSamplerAddressMode address = VK_SAMPLER_ADDRESS_MODE_REPEAT,
  bool compare = false) -> sampler_owner
{
  VkSamplerCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  info.magFilter = VK_FILTER_LINEAR;
  info.minFilter = VK_FILTER_LINEAR;
  info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  info.addressModeU = address;
  info.addressModeV = address;
  info.addressModeW = address;
  info.maxLod = 1.0F;
  info.compareEnable = compare ? VK_TRUE : VK_FALSE;
  info.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
  VkSampler sampler{ VK_NULL_HANDLE };
  if (vkCreateSampler(device, &info, nullptr, &sampler) != VK_SUCCESS) { fail_check("vkCreateSampler failed"); }
  return sampler_owner{ device, sampler };
}

}// namespace vkexec::examples

#endif
