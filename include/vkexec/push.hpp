#ifndef VKEXEC_PUSH_HPP
#define VKEXEC_PUSH_HPP

#include <vkexec/pipeline.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <type_traits>

namespace vkexec {

/// Host (non-traced) push constants: blob `T` onto the command buffer.
/// Separate from `edsl::push_constant<T>::get<&...>()` used by `bulk()` / JIT `compute_pass`.
inline auto upload_push_constants(VkCommandBuffer cmd, VkPipelineLayout layout, void const *data, std::uint32_t bytes)
  -> void
{
  if (data == nullptr || bytes == 0U) { return; }
  vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes, data);
}

template<typename T> auto upload_push_constants(VkCommandBuffer cmd, VkPipelineLayout layout, T const &params) -> void
{
  static_assert(std::is_trivially_copyable_v<T>);
  upload_push_constants(cmd, layout, &params, static_cast<std::uint32_t>(sizeof(T)));
}

template<typename T>
auto upload_push_constants(VkCommandBuffer cmd, pipeline_resources const &pipe, T const &params) -> void
{ upload_push_constants(cmd, pipe.pipeline_layout, params); }

}// namespace vkexec

#endif// VKEXEC_PUSH_HPP
