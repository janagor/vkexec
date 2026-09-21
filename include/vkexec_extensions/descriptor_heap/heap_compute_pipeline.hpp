#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_COMPUTE_PIPELINE_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_COMPUTE_PIPELINE_HPP

//! \file
//! Bindless compute pipelines (`VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT`).

#include <vkexec/compute_pipeline.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec_extensions/descriptor_heap/strategy.hpp>

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace vkexec {

/**
 * Layout for bindless null-layout compute pipelines.
 *
 * No descriptor sets or classic push constants; parameters use `cmd_push_data`.
 *
 * @see create_compute_resources, compute_pipeline
 */
struct heap_layout_desc
{
  std::vector<std::uint32_t> specialization;
  std::array<std::uint32_t, 3> local_size{ k_default_local_size };
};

/**
 * Creates bindless heap compute Vulkan objects from SPIR-V.
 *
 * Returns a `pipeline_resources` bag with null `set_layout` / `pipeline_layout` /
 * `descriptor_pool`. Do not call classic `bind_storage` on these bags. Caller owns
 * the handles and must call `destroy_compute_resources`.
 */
[[nodiscard]] auto create_compute_resources(descriptor_heap_t strategy,
  context &ctx,
  std::span<std::uint32_t const> spirv,
  heap_layout_desc const &desc) -> result<pipeline_resources>;

/**
 * Compiles `glsl` then creates bindless heap compute Vulkan objects.
 *
 * @param name Debug name for the compiler.
 */
[[nodiscard]] auto create_compute_resources(descriptor_heap_t strategy,
  context &ctx,
  std::string_view glsl,
  heap_layout_desc const &desc,
  std::string_view name = "heap.comp") -> result<pipeline_resources>;

//! Owning factory customization used by `factory::compute_pipeline(descriptor_heap, ...)`.
[[nodiscard]] auto create_compute_pipeline(descriptor_heap_t strategy,
  context &ctx,
  std::span<std::uint32_t const> spirv,
  heap_layout_desc const &desc) -> sender<compute_pipeline>;

//! Owning GLSL factory customization used by `factory::compute_pipeline(descriptor_heap, ...)`.
[[nodiscard]] auto create_compute_pipeline(descriptor_heap_t strategy,
  context &ctx,
  std::string_view glsl,
  heap_layout_desc const &desc,
  std::string_view name = "heap.comp") -> sender<compute_pipeline>;

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_COMPUTE_PIPELINE_HPP
