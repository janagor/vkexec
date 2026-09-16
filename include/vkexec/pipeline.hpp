#ifndef VKEXEC_PIPELINE_HPP
#define VKEXEC_PIPELINE_HPP

//! \file
//! Shared layout types and borrowable pipeline handle bags for classic compute.

#include <vulkan/vulkan.h>

#include <vkexec/result.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
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
 * `binding` is the descriptor binding index written into the set; `byte_size` is
 * the range (often the full buffer). Callers must set `binding` explicitly.
 */
struct storage_binding
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  VkDeviceSize byte_size{ 0 };
  std::uint32_t binding{ 0 };
};

/**
 * Writes storage-buffer descriptors for `buffers` into `set`.
 *
 * Each entry uses `storage_binding::binding` as `dstBinding`.
 */
auto write_storage_descriptors(VkDevice device, VkDescriptorSet set, std::span<storage_binding const> buffers)
  -> void;

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
 * Vulkan objects for one classic compute pipeline.
 *
 * Non-owning handle bag: fill via `create_compute_resources` or an embedder's
 * own objects. Destroy with `destroy_compute_resources` (or use Layer 2
 * `compute_pipeline`). Do not destroy individual handles while this struct is
 * still considered live. `descriptor_pool` is used to allocate sets matching
 * `set_layout` (classic pipelines only).
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

class context;

/**
 * Creates classic compute Vulkan objects from SPIR-V.
 *
 * Caller owns the returned handles and must call `destroy_compute_resources`.
 */
[[nodiscard]] auto create_compute_resources(context &ctx,
  std::span<std::uint32_t const> spirv,
  layout_desc const &desc) -> result<pipeline_resources>;

/**
 * Compiles `glsl` to SPIR-V then creates classic compute Vulkan objects.
 *
 * Caller owns the returned handles and must call `destroy_compute_resources`.
 */
[[nodiscard]] auto create_compute_resources(context &ctx,
  std::string_view glsl,
  layout_desc const &desc,
  std::string_view name = "vkexec.comp") -> result<pipeline_resources>;

//! Destroys handles in `resources` and resets them to null.
auto destroy_compute_resources(context const &ctx, pipeline_resources &resources) noexcept -> void;

}// namespace vkexec

#endif// VKEXEC_PIPELINE_HPP
