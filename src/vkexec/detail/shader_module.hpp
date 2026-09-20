#ifndef VKEXEC_DETAIL_SHADER_MODULE_HPP
#define VKEXEC_DETAIL_SHADER_MODULE_HPP

#include <vkexec/error_helpers.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <span>

namespace vkexec::detail {

[[nodiscard]] inline auto create_shader_module(VkDevice device, std::span<std::uint32_t const> spirv)
  -> result<VkShaderModule>
{
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = spirv.size_bytes();
  create_info.pCode = spirv.data();
  VkShaderModule module{ VK_NULL_HANDLE };
  if (VkResult const result = vkCreateShaderModule(device, &create_info, nullptr, &module); result != VK_SUCCESS) {
    return fail(result, "vkCreateShaderModule failed");
  }
  return module;
}

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_SHADER_MODULE_HPP
