#ifndef VKEXEC_COMPUTE_PIPELINE_HPP
#define VKEXEC_COMPUTE_PIPELINE_HPP

//! \file
//! Owning compute pipelines and helpers that build `compute_pass` closures.

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/push.hpp>
#include <vkexec/sender.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace vkexec {

class compute_pipeline;

namespace factory {

  struct make_compute_pipeline_t
  {

    /**
     * Creates a pipeline from existing SPIR-V words.
     *
     * @param ctx Context whose device creates the Vulkan objects.
     * @param spirv SPIR-V words for the compute shader.
     * @param desc Descriptor and push-constant layout.
     */
    [[nodiscard]] auto operator()(context &ctx, std::span<std::uint32_t const> spirv, layout_desc const &desc) const
      -> sender<compute_pipeline>;

    /**
     * Compiles `glsl` to SPIR-V then creates a pipeline.
     *
     * @param ctx Context whose device creates the Vulkan objects.
     * @param glsl Compute shader GLSL source.
     * @param desc Descriptor and push-constant layout.
     * @param name Debug name for the compiler.
     */
    [[nodiscard]] auto operator()(context &ctx,
      std::string_view glsl,
      layout_desc const &desc,
      std::string_view name = "vkexec.comp") const -> sender<compute_pipeline>;

    //! Creates a pipeline through an extension-owned descriptor strategy tag.
    template<class Strategy, class Desc>
    [[nodiscard]] auto
      operator()(Strategy strategy, context &ctx, std::span<std::uint32_t const> spirv, Desc const &desc) const
    { return create_compute_pipeline(strategy, ctx, spirv, desc); }

    //! Compiles GLSL and creates a pipeline through an extension-owned descriptor strategy tag.
    template<class Strategy, class Desc>
    [[nodiscard]] auto operator()(Strategy strategy,
      context &ctx,
      std::string_view glsl,
      Desc const &desc,
      std::string_view name = "vkexec.comp") const
    { return create_compute_pipeline(strategy, ctx, glsl, desc, name); }
  };

  //NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_compute_pipeline_t make_compute_pipeline{};

}// namespace factory

/**
 * Move-only compute pipeline built from SPIR-V or GLSL.
 *
 * Owns shader module, layouts, pipeline, and descriptor pool. Use `bind` /
 * `allocate_set` with `compute_pass` to dispatch. Destroy only after GPU work
 * that uses this pipeline has finished.
 *
 * @see factory::make_compute_pipeline, layout_desc, compute_pass, handles::compute_pipeline
 */
class compute_pipeline
{
public:
  //! Adopts an owned pipeline resource bag.
  [[nodiscard]] static auto make(context &ctx, std::unique_ptr<handles::compute_pipeline> resources) -> compute_pipeline
  { return compute_pipeline{ &ctx, std::move(resources) }; }

  compute_pipeline(compute_pipeline const &) = delete;
  auto operator=(compute_pipeline const &) -> compute_pipeline & = delete;

  compute_pipeline(compute_pipeline &&other) noexcept
    : ctx_(std::exchange(other.ctx_, nullptr)), resources_(std::move(other.resources_))
  {}

  auto operator=(compute_pipeline &&other) noexcept -> compute_pipeline &
  {
    if (this != &other) {
      reset();
      ctx_ = std::exchange(other.ctx_, nullptr);
      resources_ = std::move(other.resources_);
    }
    return *this;
  }

  ~compute_pipeline() { reset(); }

  //! Mutable owned Vulkan resources for this pipeline.
  [[nodiscard]] auto resources() noexcept -> handles::compute_pipeline & { return *resources_; }
  //! Const owned Vulkan resources for this pipeline.
  [[nodiscard]] auto resources() const noexcept -> handles::compute_pipeline const & { return *resources_; }

  //! Builds a `compute_bind` for recording with optional descriptor set.
  [[nodiscard]] auto bind(VkDescriptorSet set = VK_NULL_HANDLE) const -> compute_bind
  { return bind_compute(*resources_, set); }

  //! Returns the specialized local workgroup size as a `dispatch`.
  [[nodiscard]] auto local_size() const noexcept -> dispatch { return vkexec::local_size(*resources_); }

  //! Returns workgroup counts covering `work_count` invocations along X.
  [[nodiscard]] auto groups_for(std::uint32_t work_count) const noexcept -> dispatch
  { return vkexec::groups_for(*resources_, work_count); }

  //! Sender that allocates an empty descriptor set from this pipeline's pool.
  [[nodiscard]] auto allocate_set_sender() const -> sender<VkDescriptorSet>;
  //! Sender that writes `buffers` into `set`.
  [[nodiscard]] auto update_set_sender(VkDescriptorSet set, std::span<storage_binding const> buffers) const
    -> void_sender;

  //! Allocates an empty descriptor set from this pipeline's pool.
  [[nodiscard]] auto allocate_set() const -> result<VkDescriptorSet>;
  //! Writes `buffers` into `set` on the context device.
  [[nodiscard]] auto update_set(VkDescriptorSet set, std::span<storage_binding const> buffers) const -> status;

private:
  compute_pipeline(context *ctx, std::unique_ptr<handles::compute_pipeline> resources) noexcept
    : ctx_(ctx), resources_(std::move(resources))
  {}

  auto reset() noexcept -> void;

  context *ctx_{ nullptr };
  std::unique_ptr<handles::compute_pipeline> resources_;
};

/**
 * Non-owning pair of a `compute_pipeline` and a bound descriptor set for dispatch.
 *
 * `pipe` must outlive use of this binding.
 *
 * @see bind_storage_sender
 */
struct bound_compute_pipeline
{
  compute_pipeline const *pipe{ nullptr };
  VkDescriptorSet set{ VK_NULL_HANDLE };
};

/**
 * Allocates/updates a descriptor set for `buffers` and completes with `bound_compute_pipeline`.
 *
 * @param pipe Pipeline whose layout matches `buffers`.
 * @param buffers Storage bindings to write into the set.
 */
[[nodiscard]] auto bind_storage_sender(compute_pipeline const &pipe, std::span<storage_binding const> buffers)
  -> sender<bound_compute_pipeline>;

/**
 * Builds a prebuilt compute pass with push constants and automatic group counts.
 *
 * @param pipe Compute pipeline.
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
