#ifndef VKEXEC_PIPELINE_HPP
#define VKEXEC_PIPELINE_HPP

//! \file
//! Shared layout and resource types for cached compute pipelines.

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vkexec {

//! Storage-buffer access class used when building descriptor layouts.
enum class buffer_access : std::uint8_t { readonly, writeonly, readwrite };

//! Default local size X for compute shaders when not specialized.
inline constexpr std::uint32_t k_default_local_size_x = 64;
//! Default local size XYZ for compute shaders.
inline constexpr std::array<std::uint32_t, 3> k_default_local_size{ k_default_local_size_x, 1, 1 };

/**
 * One storage-buffer binding for descriptor updates.
 *
 * `binding` is the descriptor binding index; `byte_size` is the range written
 * into the descriptor (often the full buffer).
 */
struct storage_binding
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  VkDeviceSize byte_size{ 0 };
  std::uint32_t binding{ 0 };
};

/**
 * Descriptor and push-constant layout for a compute pipeline.
 *
 * Binding index is the position in `bindings` (0, 1, …). `specialization` and
 * `local_size` feed shader specialization constants when compiling / creating.
 */
struct layout_desc
{
  //! Binding index is the position in this list (0, 1, ...).
  std::vector<buffer_access> bindings;
  std::size_t push_constant_size{ 0 };
  std::vector<std::uint32_t> specialization;
  std::array<std::uint32_t, 3> local_size{ k_default_local_size };
};

/**
 * Cached Vulkan objects for one compute pipeline.
 *
 * Owned by the context pipeline cache; do not destroy handles directly.
 * `descriptor_pool` is used to allocate sets matching `set_layout`.
 */
struct pipeline_resources
{
  VkShaderModule shader{ VK_NULL_HANDLE };
  VkDescriptorSetLayout set_layout{ VK_NULL_HANDLE };
  VkPipelineLayout pipeline_layout{ VK_NULL_HANDLE };
  VkPipeline pipeline{ VK_NULL_HANDLE };
  VkDescriptorPool descriptor_pool{ VK_NULL_HANDLE };
  std::uint32_t binding_count{ 0 };
  std::size_t push_bytes{ 0 };
  std::array<std::uint32_t, 3> local_size{ k_default_local_size };
};

}// namespace vkexec

#endif// VKEXEC_PIPELINE_HPP
