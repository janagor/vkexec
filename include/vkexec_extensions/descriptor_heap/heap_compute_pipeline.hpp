#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_COMPUTE_PIPELINE_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_COMPUTE_PIPELINE_HPP

//! \file
//! Bindless compute pipelines (`VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT`).

#include <vkexec/compute_pipeline.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace vkexec {

/**
 * Layout for bindless null-layout compute pipelines.
 *
 * No descriptor sets or classic push constants; parameters use `cmd_push_data`.
 *
 * @see create_heap_compute_resources, heap_compute_pipeline
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
 * the handles and must call `destroy_heap_compute_resources`.
 */
[[nodiscard]] auto create_heap_compute_resources(context &ctx,
  std::span<std::uint32_t const> spirv,
  heap_layout_desc const &desc) -> result<pipeline_resources>;

/**
 * Compiles `glsl` then creates bindless heap compute Vulkan objects.
 *
 * @param name Debug name for the compiler.
 */
[[nodiscard]] auto create_heap_compute_resources(context &ctx,
  std::string_view glsl,
  heap_layout_desc const &desc,
  std::string_view name = "heap.comp") -> result<pipeline_resources>;

//! Destroys heap pipeline/shader handles in `resources` and resets them to null.
auto destroy_heap_compute_resources(context const &ctx, pipeline_resources &resources) noexcept -> void;

//! Builds a bindless `compute_bind` (null layout and descriptor set).
[[nodiscard]] inline auto bind_heap(pipeline_resources const &pipe) -> compute_bind
{ return compute_bind{ .pipeline = pipe.pipeline, .layout = VK_NULL_HANDLE, .set = VK_NULL_HANDLE }; }

/**
 * Thin owning wrapper over heap `pipeline_resources`.
 *
 * Bind returns a `compute_bind` with null layout/set for use with
 * `compute_heap_pass` / `record_heap_pass`.
 *
 * @see create_heap_compute_resources, compute_heap_pass, heap_layout_desc
 */
class heap_compute_pipeline
{
public:
  /**
   * Creates a heap compute pipeline from SPIR-V.
   *
   * @param ctx Context whose device creates the Vulkan objects.
   * @param spirv SPIR-V words for the compute shader.
   * @param desc Specialization and local size.
   */
  [[nodiscard]] static auto create(context &ctx, std::span<std::uint32_t const> spirv, heap_layout_desc const &desc)
    -> detail::sync_sender_fn<heap_compute_pipeline>;

  /**
   * Compiles `glsl` then creates a heap compute pipeline.
   *
   * @param name Debug name for the compiler.
   */
  [[nodiscard]] static auto
    create(context &ctx, std::string_view glsl, heap_layout_desc const &desc, std::string_view name = "heap.comp")
      -> detail::sync_sender_fn<heap_compute_pipeline>;

  //! Owning factory used after `create_heap_compute_resources`.
  [[nodiscard]] static auto make(context &ctx, std::unique_ptr<pipeline_resources> resources) -> heap_compute_pipeline
  { return heap_compute_pipeline{ &ctx, std::move(resources) }; }

  heap_compute_pipeline(heap_compute_pipeline const &) = delete;
  auto operator=(heap_compute_pipeline const &) -> heap_compute_pipeline & = delete;

  heap_compute_pipeline(heap_compute_pipeline &&other) noexcept
    : ctx_(std::exchange(other.ctx_, nullptr)), resources_(std::move(other.resources_))
  {}

  auto operator=(heap_compute_pipeline &&other) noexcept -> heap_compute_pipeline &
  {
    if (this != &other) {
      reset();
      ctx_ = std::exchange(other.ctx_, nullptr);
      resources_ = std::move(other.resources_);
    }
    return *this;
  }

  ~heap_compute_pipeline() { reset(); }

  //! Const owned Vulkan resources for this pipeline.
  [[nodiscard]] auto resources() const noexcept -> pipeline_resources const & { return *resources_; }

  //! Builds a bindless `compute_bind` (null layout and descriptor set).
  [[nodiscard]] auto bind() const -> compute_bind { return bind_heap(*resources_); }

  //! Returns workgroup counts covering `work_count` invocations along X.
  [[nodiscard]] auto groups_for(std::uint32_t work_count) const noexcept -> dispatch
  { return dispatch_groups_for(work_count, resources_->local_size.at(0)); }

private:
  heap_compute_pipeline(context *ctx, std::unique_ptr<pipeline_resources> resources) noexcept
    : ctx_(ctx), resources_(std::move(resources))
  {}

  auto reset() noexcept -> void;

  context *ctx_{ nullptr };
  std::unique_ptr<pipeline_resources> resources_;
};

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_COMPUTE_PIPELINE_HPP
