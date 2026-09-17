#ifndef VKEXEC_GRAPHICS_PIPELINE_RESOURCES_HPP
#define VKEXEC_GRAPHICS_PIPELINE_RESOURCES_HPP

//! \file
//! Borrowable graphics pipeline handle bag and Layer 1 create/destroy helpers.

#include <vkexec/context.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace vkexec {

//! Default clear color components for graphics pipelines.
constexpr float k_default_clear_r = 0.08F;
constexpr float k_default_clear_g = 0.09F;
constexpr float k_default_clear_b = 0.12F;
constexpr float k_default_clear_a = 1.0F;
constexpr float k_depth_clear_value = 1.0F;
constexpr std::uint32_t k_graphics_clear_count = 2;
constexpr std::uint32_t k_stencil_clear_value = 0;

/**
 * Configuration for classic graphics pipelines.
 *
 * When `use_mesh_vertices` is true, the pipeline expects `mesh_vertex` attributes.
 */
struct graphics_pipeline_config
{
  VkPrimitiveTopology topology{ VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
  bool alpha_blend{ false };
  float clear_r{ k_default_clear_r };
  float clear_g{ k_default_clear_g };
  float clear_b{ k_default_clear_b };
  float clear_a{ k_default_clear_a };
  bool depth_test{ false };
  bool depth_write{ true };
  bool use_mesh_vertices{ false };
};

//! Builds color + depth clear values from `cfg`.
[[nodiscard]] inline auto make_clear_values(graphics_pipeline_config const &cfg)
  -> std::array<VkClearValue, k_graphics_clear_count>
{
  std::array<VkClearValue, k_graphics_clear_count> clears{};
  clears.at(0).color = { { cfg.clear_r, cfg.clear_g, cfg.clear_b, cfg.clear_a } };
  clears.at(1).depthStencil = { .depth = k_depth_clear_value, .stencil = k_stencil_clear_value };
  return clears;
}

/**
 * Vulkan objects for one classic graphics pipeline.
 *
 * Non-owning handle bag: fill via `create_graphics_resources` or an embedder's
 * own objects. Destroy with `destroy_graphics_resources` (or Layer 2
 * `graphics_pipeline`).
 *
 * `descriptor_pool` backs set loans when `binding_count > 0`. Free unused sets
 * with `free_graphics_set`, or destroy the pool via `destroy_graphics_resources`.
 * Rebinding every frame without freeing will exhaust the pool.
 */
struct graphics_pipeline_resources
{
  graphics_pipeline_config cfg{};
  VkDescriptorSetLayout set_layout{ VK_NULL_HANDLE };
  VkPipelineLayout pipeline_layout{ VK_NULL_HANDLE };
  VkPipeline pipeline{ VK_NULL_HANDLE };
  VkDescriptorPool descriptor_pool{ VK_NULL_HANDLE };
  std::uint32_t binding_count{ 0 };
};

/**
 * Creates classic graphics Vulkan objects from SPIR-V.
 *
 * Does not allocate descriptor sets. When `storage_binding_count > 0`, creates a
 * set layout with bindings `0 .. count-1` and a descriptor pool for loans.
 * Caller owns the returned handles and must call `destroy_graphics_resources`.
 */
[[nodiscard]] auto create_graphics_resources(context &ctx,
  VkRenderPass render_pass,
  graphics_pipeline_config cfg,
  std::span<std::uint32_t const> vertex_spirv,
  std::span<std::uint32_t const> fragment_spirv,
  std::uint32_t storage_binding_count = 0) -> result<graphics_pipeline_resources>;

/**
 * Compiles GLSL to SPIR-V then creates classic graphics Vulkan objects.
 */
[[nodiscard]] auto create_graphics_resources(context &ctx,
  VkRenderPass render_pass,
  graphics_pipeline_config cfg,
  std::string_view vertex_glsl,
  std::string_view fragment_glsl,
  std::uint32_t storage_binding_count = 0) -> result<graphics_pipeline_resources>;

//! Destroys handles in `resources` and resets them to null.
auto destroy_graphics_resources(context const &ctx, graphics_pipeline_resources &resources) noexcept -> void;

/**
 * Non-owning pair of graphics pipeline resources and a bound descriptor set.
 *
 * `pipe` must outlive use of this binding.
 */
struct bound_graphics
{
  graphics_pipeline_resources const *pipe{ nullptr };
  VkDescriptorSet set{ VK_NULL_HANDLE };
};

//! Allocates an empty descriptor set from `pipe.descriptor_pool`.
//!
//! Return the set with `free_graphics_set` when finished, or free all sets by
//! destroying the pool via `destroy_graphics_resources`.
[[nodiscard]] auto allocate_graphics_set(context const &ctx, graphics_pipeline_resources const &pipe)
  -> result<VkDescriptorSet>;

/**
 * Allocates a set and writes `buffers` into it.
 *
 * The returned set is loaned from `pipe.descriptor_pool`. Call `free_graphics_set`
 * after GPU work that uses it has finished if you will allocate again.
 */
[[nodiscard]] auto bind_graphics_storage(context &ctx,
  graphics_pipeline_resources const &pipe,
  std::span<storage_binding const> buffers) -> result<bound_graphics>;

/**
 * Returns `set` to `pipe.descriptor_pool`.
 *
 * No-op when `set` or the pool is null. Do not free a set still referenced by
 * in-flight command buffers.
 */
auto free_graphics_set(context const &ctx, graphics_pipeline_resources const &pipe, VkDescriptorSet set) noexcept
  -> void;

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_PIPELINE_RESOURCES_HPP
