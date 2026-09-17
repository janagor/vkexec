#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_GRAPHICS_PIPELINE_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_GRAPHICS_PIPELINE_HPP

//! \file
//! Bindless graphics pipelines (`DESCRIPTOR_HEAP_BIT_EXT` + dynamic-rendering formats).

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace vkexec {

/**
 * Color-blend mode for heap graphics pipelines.
 *
 * `premultiplied` uses ONE / ONE_MINUS_SRC_ALPHA (vkgsplat-style over).
 */
enum class blend_mode : std::uint8_t { none, alpha, premultiplied };

/**
 * Layout for bindless null-layout graphics pipelines (dynamic rendering).
 *
 * No render pass and no descriptor sets. Color formats feed
 * `VkPipelineRenderingCreateInfo`. Depth is optional (`UNDEFINED` = none).
 *
 * @see create_heap_graphics_resources, heap_graphics_pipeline
 */
struct heap_graphics_layout_desc
{
  VkPrimitiveTopology topology{ VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
  blend_mode blend{ blend_mode::none };
  bool depth_test{ false };
  bool depth_write{ true };
  std::vector<VkFormat> color_formats;
  VkFormat depth_format{ VK_FORMAT_UNDEFINED };
};

/**
 * Creates bindless heap graphics Vulkan objects from vertex/fragment SPIR-V.
 *
 * Returns a `pipeline_resources` bag with null layouts/pool and a live pipeline.
 * Shader modules are destroyed after pipeline creation. Caller owns the bag and
 * must call `destroy_heap_graphics_resources`.
 *
 * Requires `VK_EXT_descriptor_heap` and a Vulkan 1.3+ device (dynamic rendering
 * pipeline create info).
 */
[[nodiscard]] auto create_heap_graphics_resources(context &ctx,
  std::span<std::uint32_t const> vertex_spirv,
  std::span<std::uint32_t const> fragment_spirv,
  heap_graphics_layout_desc const &desc) -> result<pipeline_resources>;

/**
 * Compiles GLSL then creates bindless heap graphics Vulkan objects.
 *
 * @param vertex_name Debug name for the vertex shader compiler.
 * @param fragment_name Debug name for the fragment shader compiler.
 */
[[nodiscard]] auto create_heap_graphics_resources(context &ctx,
  std::string_view vertex_glsl,
  std::string_view fragment_glsl,
  heap_graphics_layout_desc const &desc,
  std::string_view vertex_name = "heap.vert",
  std::string_view fragment_name = "heap.frag") -> result<pipeline_resources>;

//! Destroys the heap graphics pipeline handle in `resources` and resets the bag.
auto destroy_heap_graphics_resources(context const &ctx, pipeline_resources &resources) noexcept -> void;

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_GRAPHICS_PIPELINE_HPP
