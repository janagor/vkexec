#ifndef VKEXEC_GRAPHICS_GRAPHICS_HPP
#define VKEXEC_GRAPHICS_GRAPHICS_HPP

//! \file
//! Layer 2 owning graphics pipelines for swapchain render passes.

#include <vkexec/context.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace vkexec {

class mesh;

/**
 * Graphics pipeline built from precompiled SPIR-V or GLSL source strings.
 *
 * Thin Layer 2 owner over `graphics_pipeline_resources` plus an optional retained
 * descriptor set for storage buffers passed at create time. Compatible with a
 * `window` render pass. Use `draw` / `record_draw` inside a begun frame.
 *
 * @see window, mesh, draw, create_graphics_resources
 */
class graphics_pipeline
{
public:
  /**
   * Creates a graphics pipeline from SPIR-V with an explicit config.
   *
   * @param ctx Context that owns the device.
   * @param render_pass Compatible render pass (typically from `window`).
   * @param cfg Topology, blending, clears, depth, mesh vertex layout.
   * @param vertex_spirv Vertex shader SPIR-V.
   * @param fragment_spirv Fragment shader SPIR-V.
   * @param buffers Optional storage buffers bound as descriptors.
   */
  [[nodiscard]] static auto create(context &ctx,
    VkRenderPass render_pass,
    graphics_pipeline_config cfg,
    std::span<std::uint32_t const> vertex_spirv,
    std::span<std::uint32_t const> fragment_spirv,
    std::span<storage_binding const> buffers = {}) -> detail::sync_sender_fn<graphics_pipeline>;

  //! Creates a graphics pipeline from SPIR-V with default config.
  [[nodiscard]] static auto create(context &ctx,
    VkRenderPass render_pass,
    std::span<std::uint32_t const> vertex_spirv,
    std::span<std::uint32_t const> fragment_spirv,
    std::span<storage_binding const> buffers = {}) -> detail::sync_sender_fn<graphics_pipeline>;

  //! Creates a graphics pipeline by compiling GLSL with an explicit config.
  [[nodiscard]] static auto create(context &ctx,
    VkRenderPass render_pass,
    graphics_pipeline_config cfg,
    std::string_view vertex_glsl,
    std::string_view fragment_glsl,
    std::span<storage_binding const> buffers = {}) -> detail::sync_sender_fn<graphics_pipeline>;

  //! Creates a graphics pipeline by compiling GLSL with default config.
  [[nodiscard]] static auto create(context &ctx,
    VkRenderPass render_pass,
    std::string_view vertex_glsl,
    std::string_view fragment_glsl,
    std::span<storage_binding const> buffers = {}) -> detail::sync_sender_fn<graphics_pipeline>;

  //! Layer 2 factory used after `create_graphics_resources` + optional set install.
  [[nodiscard]] static auto make(context &ctx,
    std::unique_ptr<graphics_pipeline_resources> resources,
    VkDescriptorSet set = VK_NULL_HANDLE,
    std::vector<storage_binding> buffers = {}) -> graphics_pipeline
  {
    graphics_pipeline pipe{ &ctx, std::move(resources) };
    pipe.descriptor_set_ = set;
    pipe.buffers_ = std::move(buffers);
    return pipe;
  }

  ~graphics_pipeline() { reset(); }

  graphics_pipeline(graphics_pipeline const &) = delete;
  auto operator=(graphics_pipeline const &) -> graphics_pipeline & = delete;

  graphics_pipeline(graphics_pipeline &&other) noexcept
    : ctx_(std::exchange(other.ctx_, nullptr)), resources_(std::move(other.resources_)),
      descriptor_set_(std::exchange(other.descriptor_set_, VK_NULL_HANDLE)), buffers_(std::move(other.buffers_))
  {}

  auto operator=(graphics_pipeline &&other) noexcept -> graphics_pipeline &
  {
    if (this == &other) { return *this; }
    reset();
    ctx_ = std::exchange(other.ctx_, nullptr);
    resources_ = std::move(other.resources_);
    descriptor_set_ = std::exchange(other.descriptor_set_, VK_NULL_HANDLE);
    buffers_ = std::move(other.buffers_);
    return *this;
  }

  //! Mutable owned Vulkan resources for this pipeline.
  [[nodiscard]] auto resources() noexcept -> graphics_pipeline_resources & { return *resources_; }
  //! Const owned Vulkan resources for this pipeline.
  [[nodiscard]] auto resources() const noexcept -> graphics_pipeline_resources const & { return *resources_; }

  //! Vulkan pipeline handle.
  [[nodiscard]] auto pipeline() const noexcept -> VkPipeline { return resources_->pipeline; }
  //! Config used at creation.
  [[nodiscard]] auto config() const noexcept -> graphics_pipeline_config const & { return resources_->cfg; }

  //! Builds a `graphics_bind` for recording with the retained descriptor set.
  [[nodiscard]] auto bind() const -> graphics_bind { return bind_graphics(*resources_, descriptor_set_); }

  /**
   * Records viewport/scissor, bind, and a non-indexed draw into an open render pass.
   */
  auto record_draw(VkCommandBuffer cmd, VkExtent2D extent, std::uint32_t vertex_count) const -> void;

  //! Records a mesh draw (vertex/index binds + indexed draw) into an open render pass.
  auto record_draw(VkCommandBuffer cmd, VkExtent2D extent, mesh const &drawn) const -> void;

  /**
   * Begins a render pass, records a non-indexed draw, and ends the pass.
   */
  auto draw(VkCommandBuffer cmd,
    VkRenderPass render_pass,
    VkFramebuffer framebuffer,
    VkExtent2D extent,
    std::uint32_t vertex_count) const -> status;

  //! Begins a render pass, records a mesh draw, and ends the pass.
  auto draw(VkCommandBuffer cmd,
    VkRenderPass render_pass,
    VkFramebuffer framebuffer,
    VkExtent2D extent,
    mesh const &drawn) const -> status;

private:
  graphics_pipeline(context *ctx, std::unique_ptr<graphics_pipeline_resources> resources) noexcept
    : ctx_(ctx), resources_(std::move(resources))
  {}

  auto reset() noexcept -> void;

  context *ctx_{ nullptr };
  std::unique_ptr<graphics_pipeline_resources> resources_;
  VkDescriptorSet descriptor_set_{ VK_NULL_HANDLE };
  std::vector<storage_binding> buffers_;
};

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_GRAPHICS_HPP
