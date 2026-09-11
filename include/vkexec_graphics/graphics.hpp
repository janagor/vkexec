#ifndef VKEXEC_GRAPHICS_GRAPHICS_HPP
#define VKEXEC_GRAPHICS_GRAPHICS_HPP

//! \file
//! Graphics pipelines for swapchain render passes (SPIR-V or GLSL).

#include <vkexec/context.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace vkexec {

class mesh;

//! Default clear color components for graphics pipelines.
constexpr float k_default_clear_r = 0.08F;
constexpr float k_default_clear_g = 0.09F;
constexpr float k_default_clear_b = 0.12F;
constexpr float k_default_clear_a = 1.0F;
constexpr float k_depth_clear_value = 1.0F;
constexpr std::uint32_t k_graphics_clear_count = 2;
constexpr std::uint32_t k_stencil_clear_value = 0;

/**
 * Configuration for `graphics_pipeline::create`.
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
 * Graphics pipeline built from precompiled SPIR-V or GLSL source strings.
 *
 * Owns pipeline, layout, and optional storage-buffer descriptor set. Compatible
 * with a `window` render pass. Use `draw` / `record_draw` inside a begun frame.
 *
 * @see window, mesh, draw
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

  ~graphics_pipeline() { destroy(); }

  graphics_pipeline(graphics_pipeline const &) = delete;
  auto operator=(graphics_pipeline const &) -> graphics_pipeline & = delete;

  graphics_pipeline(graphics_pipeline &&other) noexcept
    : device_(other.device_), cfg_(other.cfg_), layout_(other.layout_), pipeline_(other.pipeline_),
      set_layout_(other.set_layout_), descriptor_pool_(other.descriptor_pool_), descriptor_set_(other.descriptor_set_),
      buffers_(std::move(other.buffers_))
  { other.release(); }

  auto operator=(graphics_pipeline &&other) noexcept -> graphics_pipeline &
  {
    if (this == &other) { return *this; }
    destroy();
    device_ = other.device_;
    cfg_ = other.cfg_;
    layout_ = other.layout_;
    pipeline_ = other.pipeline_;
    set_layout_ = other.set_layout_;
    descriptor_pool_ = other.descriptor_pool_;
    descriptor_set_ = other.descriptor_set_;
    buffers_ = std::move(other.buffers_);
    other.release();
    return *this;
  }

  //! Vulkan pipeline handle.
  [[nodiscard]] auto pipeline() const noexcept -> VkPipeline { return pipeline_; }
  //! Config used at creation.
  [[nodiscard]] auto config() const noexcept -> graphics_pipeline_config const & { return cfg_; }

  /**
   * Records viewport/scissor, bind, and a non-indexed draw into an open render pass.
   *
   * @param cmd Command buffer currently inside a render pass.
   * @param extent Framebuffer extent for viewport/scissor.
   * @param vertex_count Vertex count for `vkCmdDraw`.
   */
  auto record_draw(VkCommandBuffer cmd, VkExtent2D extent, std::uint32_t vertex_count) const -> void;

  //! Records a mesh draw (vertex/index binds + indexed draw) into an open render pass.
  auto record_draw(VkCommandBuffer cmd, VkExtent2D extent, mesh const &drawn) const -> void;

  /**
   * Begins a render pass, records a non-indexed draw, and ends the pass.
   *
   * @param render_pass Compatible render pass.
   * @param framebuffer Target framebuffer.
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
  struct bound_buffer
  {
    std::uint32_t binding{ 0 };
    VkBuffer buffer{ VK_NULL_HANDLE };
    VkDeviceSize byte_size{ 0 };
  };

  graphics_pipeline() = default;

  auto destroy() noexcept -> void;

  auto release() noexcept -> void;

  auto complete(context &ctx,
    VkRenderPass render_pass,
    std::vector<std::uint32_t> const &vs_spv,
    std::vector<std::uint32_t> const &fs_spv,
    std::span<storage_binding const> buffers) -> status;

  [[nodiscard]] auto create_module(std::vector<std::uint32_t> const &spirv) const -> result<VkShaderModule>;

  auto bind_draw_state(VkCommandBuffer cmd, VkExtent2D extent) const -> void;

  auto begin_pass(VkCommandBuffer cmd, VkRenderPass render_pass, VkFramebuffer framebuffer, VkExtent2D extent) const
    -> void;

  VkDevice device_{ VK_NULL_HANDLE };
  graphics_pipeline_config cfg_{};
  VkPipelineLayout layout_{ VK_NULL_HANDLE };
  VkPipeline pipeline_{ VK_NULL_HANDLE };
  VkDescriptorSetLayout set_layout_{ VK_NULL_HANDLE };
  VkDescriptorPool descriptor_pool_{ VK_NULL_HANDLE };
  VkDescriptorSet descriptor_set_{ VK_NULL_HANDLE };
  std::vector<bound_buffer> buffers_;
};

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_GRAPHICS_HPP
