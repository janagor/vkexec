#include <vkexec_tools/spirv_compile.hpp>

#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/heap_graphics_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/strategy.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vkexec {

auto create_compute_resources(context &ctx, std::string_view glsl, layout_desc const &desc, std::string_view name)
  -> result<pipeline_resources>
{
  if (glsl.empty()) { return fail(errc::invalid_argument, "create_compute_resources requires non-empty GLSL"); }
  VKEXEC_TRY_ASSIGN(spirv, compile_glsl_to_spirv(glsl, name, shader_kind::compute, ctx.api_version()));
  return create_compute_resources(ctx, spirv, desc);
}

auto factory::make_compute_pipeline_t::operator()(::vkexec::context &ctx,
  std::string_view glsl,
  layout_desc const &desc,
  std::string_view name) const -> sender<::vkexec::compute_pipeline>
{
  return make_sender<::vkexec::compute_pipeline>(
    [&ctx, glsl = std::string(glsl), desc, name = std::string(name)]() -> result<::vkexec::compute_pipeline> {
      VKEXEC_TRY_ASSIGN(owned, create_compute_resources(ctx, glsl, desc, name));
      return compute_pipeline::make(ctx, std::make_unique<pipeline_resources>(owned));
    });
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
auto create_graphics_resources(context &ctx,
  VkRenderPass render_pass,
  graphics_pipeline_config cfg,
  std::string_view vertex_glsl,
  std::string_view fragment_glsl,
  std::uint32_t storage_binding_count) -> result<graphics_pipeline_resources>
// NOLINTEND(bugprone-easily-swappable-parameters)
{
  if (vertex_glsl.empty() || fragment_glsl.empty()) {
    return fail(errc::invalid_argument, "create_graphics_resources requires non-empty GLSL");
  }
  VKEXEC_TRY_ASSIGN(
    vert_spirv, compile_glsl_to_spirv(vertex_glsl, "vkexec.vert", shader_kind::vertex, ctx.api_version()));
  VKEXEC_TRY_ASSIGN(
    frag_spirv, compile_glsl_to_spirv(fragment_glsl, "vkexec.frag", shader_kind::fragment, ctx.api_version()));
  return create_graphics_resources(ctx, render_pass, cfg, vert_spirv, frag_spirv, storage_binding_count);
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
auto factory::make_graphics_pipeline_t::operator()(::vkexec::context &ctx,
  VkRenderPass render_pass,
  graphics_pipeline_config cfg,
  std::string_view vertex_glsl,
  std::string_view fragment_glsl,
  std::span<storage_binding const> buffers) const -> sender<::vkexec::graphics_pipeline>
// NOLINTEND(bugprone-easily-swappable-parameters)
{
  return make_sender<::vkexec::graphics_pipeline>(
    [&ctx,
      render_pass,
      cfg,
      vertex_glsl = std::string(vertex_glsl),
      fragment_glsl = std::string(fragment_glsl),
      owned = std::vector(buffers.begin(), buffers.end())]() mutable -> result<::vkexec::graphics_pipeline> {
      VKEXEC_TRY_ASSIGN(gfx_resources,
        create_graphics_resources(
          ctx, render_pass, cfg, vertex_glsl, fragment_glsl, static_cast<std::uint32_t>(owned.size())));
      VkDescriptorSet set = VK_NULL_HANDLE;
      if (!owned.empty()) {
        auto bound = bind_graphics_storage(ctx, gfx_resources, owned);
        if (!bound) {
          destroy_graphics_resources(ctx, gfx_resources);
          return fail(bound);
        }
        set = bound->set;
      }
      return graphics_pipeline::make(
        ctx, std::make_unique<graphics_pipeline_resources>(gfx_resources), set, std::move(owned));
    });
}

auto factory::make_graphics_pipeline_t::operator()(::vkexec::context &ctx,
  VkRenderPass render_pass,
  std::string_view vertex_glsl,
  std::string_view fragment_glsl,
  std::span<storage_binding const> buffers) const -> sender<::vkexec::graphics_pipeline>
{
  return factory::make_graphics_pipeline(
    ctx, render_pass, graphics_pipeline_config{}, vertex_glsl, fragment_glsl, buffers);
}

auto create_compute_resources(descriptor_heap_t strategy,
  context &ctx,
  std::string_view glsl,
  heap_layout_desc const &desc,
  std::string_view name) -> result<pipeline_resources>
{
  if (glsl.empty()) { return fail(errc::invalid_argument, "create_compute_resources requires non-empty GLSL"); }
  VKEXEC_TRY_ASSIGN(spirv, compile_glsl_to_spirv(glsl, name, shader_kind::compute, ctx.api_version()));
  return create_compute_resources(strategy, ctx, spirv, desc);
}

auto create_compute_pipeline(descriptor_heap_t strategy,
  context &ctx,
  std::string_view glsl,
  heap_layout_desc const &desc,
  std::string_view name) -> sender<::vkexec::compute_pipeline>
{
  return make_sender<::vkexec::compute_pipeline>(
    [&ctx, strategy, glsl = std::string(glsl), desc, name = std::string(name)]() -> result<::vkexec::compute_pipeline> {
      VKEXEC_TRY_ASSIGN(owned, create_compute_resources(strategy, ctx, glsl, desc, name));
      return compute_pipeline::make(ctx, std::make_unique<pipeline_resources>(owned));
    });
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
auto create_graphics_resources(descriptor_heap_t strategy,
  context &ctx,
  std::string_view vertex_glsl,
  std::string_view fragment_glsl,
  heap_graphics_layout_desc const &desc,
  std::string_view vertex_name,
  std::string_view fragment_name) -> result<pipeline_resources>
// NOLINTEND(bugprone-easily-swappable-parameters)
{
  if (vertex_glsl.empty() || fragment_glsl.empty()) {
    return fail(errc::invalid_argument, "create_graphics_resources requires non-empty GLSL");
  }
  VKEXEC_TRY_ASSIGN(
    vert_spirv, compile_glsl_to_spirv(vertex_glsl, vertex_name, shader_kind::vertex, ctx.api_version()));
  VKEXEC_TRY_ASSIGN(
    frag_spirv, compile_glsl_to_spirv(fragment_glsl, fragment_name, shader_kind::fragment, ctx.api_version()));
  return create_graphics_resources(strategy, ctx, vert_spirv, frag_spirv, desc);
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
auto factory::make_descriptor_graphics_pipeline_t::operator()(::vkexec::context &ctx,
  std::string_view vertex_glsl,
  std::string_view fragment_glsl,
  heap_graphics_layout_desc const &desc,
  std::string_view vertex_name,
  std::string_view fragment_name) const -> sender<::vkexec::descriptor_graphics_pipeline>
// NOLINTEND(bugprone-easily-swappable-parameters)
{
  return make_sender<::vkexec::descriptor_graphics_pipeline>(
    [&ctx,
      vertex_glsl = std::string(vertex_glsl),
      fragment_glsl = std::string(fragment_glsl),
      desc,
      vertex_name = std::string(vertex_name),
      fragment_name = std::string(fragment_name)]() -> result<::vkexec::descriptor_graphics_pipeline> {
      VKEXEC_TRY_ASSIGN(owned,
        create_graphics_resources(descriptor_heap, ctx, vertex_glsl, fragment_glsl, desc, vertex_name, fragment_name));
      return ::vkexec::descriptor_graphics_pipeline::make(ctx, std::make_unique<pipeline_resources>(owned));
    });
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
auto create_graphics_pipeline([[maybe_unused]] descriptor_heap_t strategy,
  context &ctx,
  std::string_view vertex_glsl,
  std::string_view fragment_glsl,
  heap_graphics_layout_desc const &desc,
  std::string_view vertex_name,
  std::string_view fragment_name) -> sender<::vkexec::descriptor_graphics_pipeline>
// NOLINTEND(bugprone-easily-swappable-parameters)
{
  return factory::make_descriptor_graphics_pipeline(ctx, vertex_glsl, fragment_glsl, desc, vertex_name, fragment_name);
}

}// namespace vkexec
