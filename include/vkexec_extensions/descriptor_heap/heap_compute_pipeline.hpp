#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_COMPUTE_PIPELINE_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_COMPUTE_PIPELINE_HPP

//! \file
//! Cached bindless compute pipelines (`VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT`).

#include <vkexec/compute_pipeline.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>

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
 * @see heap_compute_pipeline
 */
struct heap_layout_desc
{
  std::vector<std::uint32_t> specialization;
  std::array<std::uint32_t, 3> local_size{ k_default_local_size };
};

/**
 * Cached compute pipeline created with `VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT`.
 *
 * Resources live in the context pipeline cache. Bind returns a `compute_bind` with
 * null layout/set for use with `compute_heap_pass` / `record_heap_pass`.
 *
 * @see compute_heap_pass, heap_layout_desc
 */
class heap_compute_pipeline
{
public:
  /**
   * Creates (or reuses) a heap compute pipeline from SPIR-V.
   *
   * @param ctx Context that owns the pipeline cache.
   * @param spirv SPIR-V words for the compute shader.
   * @param desc Specialization and local size.
   */
  [[nodiscard]] static auto create(context &ctx, std::span<std::uint32_t const> spirv, heap_layout_desc const &desc)
    -> detail::sync_sender_fn<heap_compute_pipeline>;

  /**
   * Compiles `glsl` then creates (or reuses) a heap compute pipeline.
   *
   * @param name Debug name for the compiler and cache key.
   */
  [[nodiscard]] static auto
    create(context &ctx, std::string_view glsl, heap_layout_desc const &desc, std::string_view name = "heap.comp")
      -> detail::sync_sender_fn<heap_compute_pipeline>;

  //! Const cached Vulkan resources for this pipeline.
  [[nodiscard]] auto resources() const noexcept -> pipeline_resources const & { return *resources_; }

  //! Builds a bindless `compute_bind` (null layout and descriptor set).
  [[nodiscard]] auto bind() const -> compute_bind
  { return compute_bind{ .pipeline = resources_->pipeline, .layout = VK_NULL_HANDLE, .set = VK_NULL_HANDLE }; }

  //! Returns workgroup counts covering `work_count` invocations along X.
  [[nodiscard]] auto groups_for(std::uint32_t work_count) const noexcept -> dispatch
  { return dispatch_groups_for(work_count, resources_->local_size.at(0)); }

private:
  heap_compute_pipeline(context * /*ctx*/, pipeline_resources *resources) noexcept : resources_(resources) {}

  pipeline_resources *resources_{ nullptr };
};

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_COMPUTE_PIPELINE_HPP
