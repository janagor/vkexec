#ifndef VKEXEC_COMPUTE_PIPELINE_HPP
#define VKEXEC_COMPUTE_PIPELINE_HPP

//! \file
//! Cached compute pipelines and helpers that build `compute_pass` closures.

#include <vkexec/context.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/push.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace vkexec {

/**
 * Handle to a cached compute pipeline built from SPIR-V or GLSL.
 *
 * Resources live in the context pipeline cache; this object is a non-owning
 * view. Use `bind` / `allocate_set` with `compute_pass` to dispatch.
 *
 * @see layout_desc, compute_pass, pipeline_resources
 */
class compute_pipeline
{
public:
  /**
   * Creates (or reuses) a cached pipeline from existing SPIR-V words.
   *
   * @param ctx Context that owns the pipeline cache.
   * @param spirv SPIR-V words for the compute shader.
   * @param desc Descriptor and push-constant layout.
   */
  [[nodiscard]] static auto create(context &ctx, std::span<std::uint32_t const> spirv, layout_desc const &desc)
    -> detail::sync_sender_fn<compute_pipeline>;

  /**
   * Compiles `glsl` to SPIR-V then creates (or reuses) a cached pipeline.
   *
   * @param ctx Context that owns the pipeline cache.
   * @param glsl Compute shader GLSL source.
   * @param desc Descriptor and push-constant layout.
   * @param name Debug name for the compiler and cache key.
   */
  [[nodiscard]] static auto
    create(context &ctx, std::string_view glsl, layout_desc const &desc, std::string_view name = "vkexec.comp")
      -> detail::sync_sender_fn<compute_pipeline>;

  //! Mutable cached Vulkan resources for this pipeline.
  [[nodiscard]] auto resources() noexcept -> pipeline_resources & { return *resources_; }
  //! Const cached Vulkan resources for this pipeline.
  [[nodiscard]] auto resources() const noexcept -> pipeline_resources const & { return *resources_; }

  //! Builds a `compute_bind` for recording with optional descriptor set.
  [[nodiscard]] auto bind(VkDescriptorSet set = VK_NULL_HANDLE) const -> compute_bind
  { return bind_compute(*resources_, set); }

  //! Returns the specialized local workgroup size as a `dispatch`.
  [[nodiscard]] auto local_size() const noexcept -> dispatch
  {
    return dispatch{
      .x = resources_->local_size.at(0), .y = resources_->local_size.at(1), .z = resources_->local_size.at(2)
    };
  }

  //! Returns workgroup counts covering `work_count` invocations along X.
  [[nodiscard]] auto groups_for(std::uint32_t work_count) const noexcept -> dispatch
  { return dispatch_groups_for(work_count, resources_->local_size.at(0)); }

  //! Sender that allocates an empty descriptor set from this pipeline's pool.
  [[nodiscard]] auto allocate_set_sender() const -> detail::sync_sender_fn<VkDescriptorSet>;
  //! Sender that writes `buffers` into `set`.
  [[nodiscard]] auto update_set_sender(VkDescriptorSet set, std::span<storage_binding const> buffers) const
    -> detail::sync_void_sender_fn;

  //! Allocates an empty descriptor set from this pipeline's pool.
  [[nodiscard]] auto allocate_set() const -> result<VkDescriptorSet>;
  //! Writes `buffers` into `set` on the context device.
  [[nodiscard]] auto update_set(VkDescriptorSet set, std::span<storage_binding const> buffers) const -> status;

private:
  compute_pipeline(context *ctx, pipeline_resources *pipe) noexcept : ctx_(ctx), resources_(pipe) {}

  context *ctx_{ nullptr };
  pipeline_resources *resources_{ nullptr };
};

/**
 * Pair of a `compute_pipeline` and a bound descriptor set for dispatch.
 *
 * @see bind_storage_sender
 */
struct bound_compute_pipeline
{
  compute_pipeline pipe;
  VkDescriptorSet set{ VK_NULL_HANDLE };
};

/**
 * Allocates/updates a descriptor set for `buffers` and completes with `bound_compute_pipeline`.
 *
 * @param pipe Pipeline whose layout matches `buffers`.
 * @param buffers Storage bindings to write into the set.
 */
[[nodiscard]] auto bind_storage_sender(compute_pipeline const &pipe, std::span<storage_binding const> buffers)
  -> detail::sync_sender_fn<bound_compute_pipeline>;

/**
 * Builds a prebuilt compute pass with push constants and automatic group counts.
 *
 * @param pipe Cached compute pipeline.
 * @param set Descriptor set matching the pipeline layout (may be null if unused).
 * @param params Trivially copyable push-constant blob.
 * @param work_count Invocation count along X (converted via `groups_for`).
 */
template<typename Params>
auto compute_pass(compute_pipeline const &pipe, VkDescriptorSet set, Params const &params, std::uint32_t work_count)
  -> prebuilt_compute_pass_closure
{ return compute_pass(pipe.bind(set), params, pipe.groups_for(work_count)); }

//! Builds a prebuilt compute pass without push constants.
auto compute_pass(compute_pipeline const &pipe, VkDescriptorSet set, std::uint32_t work_count)
  -> prebuilt_compute_pass_closure;

//! Uploads push constants using `pipe.resources().pipeline_layout`.
template<typename T>
auto upload_push_constants(VkCommandBuffer cmd, compute_pipeline const &pipe, T const &params) -> void
{ upload_push_constants(cmd, pipe.resources().pipeline_layout, params); }

}// namespace vkexec

#endif// VKEXEC_COMPUTE_PIPELINE_HPP
