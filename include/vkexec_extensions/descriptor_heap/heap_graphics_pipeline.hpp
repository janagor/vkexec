#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_GRAPHICS_PIPELINE_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_GRAPHICS_PIPELINE_HPP

//! \file
//! Bindless graphics pipelines (`DESCRIPTOR_HEAP_BIT_EXT` + dynamic-rendering formats).

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec_extensions/descriptor_heap/strategy.hpp>
#include <vkexec_graphics/graphics.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
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
 * @see create, descriptor_graphics_pipeline
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
 * Returns a `handles::graphics_pipeline` bag with null layouts/pool and a live pipeline.
 * Shader modules are destroyed after pipeline creation. Caller owns the bag and
 * must call `destroy(descriptor_heap, ...)`.
 *
 * Requires `VK_EXT_descriptor_heap` and a Vulkan 1.3+ device (dynamic rendering
 * pipeline create info).
 */
[[nodiscard]] auto create(descriptor_heap_t strategy,
  context &ctx,
  std::span<std::uint32_t const> vertex_spirv,
  std::span<std::uint32_t const> fragment_spirv,
  heap_graphics_layout_desc const &desc) -> result<handles::graphics_pipeline>;

/**
 * Compiles GLSL then creates bindless heap graphics Vulkan objects.
 *
 * @param vertex_name Debug name for the vertex shader compiler.
 * @param fragment_name Debug name for the fragment shader compiler.
 */
[[nodiscard]] auto create(descriptor_heap_t strategy,
  context &ctx,
  std::string_view vertex_glsl,
  std::string_view fragment_glsl,
  heap_graphics_layout_desc const &desc,
  std::string_view vertex_name = "heap.vert",
  std::string_view fragment_name = "heap.frag") -> result<handles::graphics_pipeline>;

//! Destroys a descriptor-heap graphics resource bag and resets it.
auto destroy(descriptor_heap_t strategy, context const &ctx, handles::graphics_pipeline &resources) noexcept
  -> void;

namespace owned {
class descriptor_graphics_pipeline;
}// namespace owned

namespace factory {

  struct make_descriptor_graphics_pipeline_t
  {

    /**
     * Creates a heap graphics pipeline from vertex/fragment SPIR-V.
     *
     * @param ctx Context whose device creates the Vulkan objects.
     * @param vertex_spirv Vertex SPIR-V words.
     * @param fragment_spirv Fragment SPIR-V words.
     * @param desc Formats, blend, and raster state.
     */
    [[nodiscard]] auto operator()(context &ctx,
      std::span<std::uint32_t const> vertex_spirv,
      std::span<std::uint32_t const> fragment_spirv,
      heap_graphics_layout_desc const &desc) const -> sender<owned::descriptor_graphics_pipeline>;

    /**
     * Compiles GLSL then creates a heap graphics pipeline.
     *
     * @param vertex_name Debug name for the vertex shader compiler.
     * @param fragment_name Debug name for the fragment shader compiler.
     */
    [[nodiscard]] auto operator()(context &ctx,
      std::string_view vertex_glsl,
      std::string_view fragment_glsl,
      heap_graphics_layout_desc const &desc,
      std::string_view vertex_name = "heap.vert",
      std::string_view fragment_name = "heap.frag") const -> sender<owned::descriptor_graphics_pipeline>;
  };

  //NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_descriptor_graphics_pipeline_t make_descriptor_graphics_pipeline{};

}// namespace factory

/**
 * Thin owning wrapper over heap graphics `handles::graphics_pipeline`.
 *
 * Bind returns a `compute_bind` with null layout/set (same bag shape as heap
 * compute) for use with `record_draw(context, ...)`.
 *
 * @see create, heap_graphics_layout_desc, factory::make_descriptor_graphics_pipeline
 */
namespace owned {

class descriptor_graphics_pipeline
{
public:
  //! Owning factory used after `create(descriptor_heap, ...)`.
  [[nodiscard]] static auto make(context &ctx, std::unique_ptr<handles::graphics_pipeline> resources)
    -> descriptor_graphics_pipeline
  { return descriptor_graphics_pipeline{ &ctx, std::move(resources) }; }

  descriptor_graphics_pipeline(descriptor_graphics_pipeline const &) = delete;
  auto operator=(descriptor_graphics_pipeline const &) -> descriptor_graphics_pipeline & = delete;

  descriptor_graphics_pipeline(descriptor_graphics_pipeline &&other) noexcept
    : ctx_(std::exchange(other.ctx_, nullptr)), resources_(std::move(other.resources_))
  {}

  auto operator=(descriptor_graphics_pipeline &&other) noexcept -> descriptor_graphics_pipeline &
  {
    if (this != &other) {
      reset();
      ctx_ = std::exchange(other.ctx_, nullptr);
      resources_ = std::move(other.resources_);
    }
    return *this;
  }

  ~descriptor_graphics_pipeline() { reset(); }

  //! Const owned Vulkan resources for this pipeline.
  [[nodiscard]] auto resources() const noexcept -> handles::graphics_pipeline const & { return *resources_; }

  //! Builds a bindless bind (null layout and descriptor set).
  [[nodiscard]] auto bind() const -> compute_bind
  { return compute_bind{ .pipeline = resources_->pipeline, .layout = resources_->pipeline_layout }; }

private:
  descriptor_graphics_pipeline(context *ctx, std::unique_ptr<handles::graphics_pipeline> resources) noexcept
    : ctx_(ctx), resources_(std::move(resources))
  {}

  auto reset() noexcept -> void;

  context *ctx_{ nullptr };
  std::unique_ptr<handles::graphics_pipeline> resources_;
};

}// namespace owned

//! Owning factory customization used by `factory::make_graphics_pipeline(descriptor_heap, ...)`.
[[nodiscard]] auto create_graphics_pipeline(descriptor_heap_t strategy,
  context &ctx,
  std::span<std::uint32_t const> vertex_spirv,
  std::span<std::uint32_t const> fragment_spirv,
  heap_graphics_layout_desc const &desc) -> sender<owned::descriptor_graphics_pipeline>;

//! Owning GLSL factory customization used by `factory::make_graphics_pipeline(descriptor_heap, ...)`.
[[nodiscard]] auto create_graphics_pipeline(descriptor_heap_t strategy,
  context &ctx,
  std::string_view vertex_glsl,
  std::string_view fragment_glsl,
  heap_graphics_layout_desc const &desc,
  std::string_view vertex_name = "heap.vert",
  std::string_view fragment_name = "heap.frag") -> sender<owned::descriptor_graphics_pipeline>;

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_GRAPHICS_PIPELINE_HPP
