#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>

#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/compute_create.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/spirv_compile.hpp>
#include <vkexec_extensions/descriptor_heap/strategy.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace vkexec {
auto create_compute_resources(descriptor_heap_t /*strategy*/,
  context &ctx,
  std::span<std::uint32_t const> spirv,
  heap_layout_desc const &desc) -> result<pipeline_resources>
{
  detail::compute_create_info const info{ .bindings = {},
    .push_bytes = 0,
    .specialization = desc.specialization,
    .local_size = desc.local_size,
    .create_binding_objects = false };
  return detail::create_compute_resources_with<detail::heap_descriptor_backend>(
    ctx, spirv, info, "create_compute_resources requires non-empty SPIR-V");
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
  std::span<std::uint32_t const> spirv,
  heap_layout_desc const &desc) -> detail::sync_sender_fn<compute_pipeline>
{
  return detail::make_sync_sender_fn<compute_pipeline>([&ctx, strategy, spirv, desc]() -> result<compute_pipeline> {
    VKEXEC_TRY_ASSIGN(owned, create_compute_resources(strategy, ctx, spirv, desc));
    return compute_pipeline::make(ctx, std::make_unique<pipeline_resources>(owned));
  });
}

auto create_compute_pipeline(descriptor_heap_t strategy,
  context &ctx,
  std::string_view glsl,
  heap_layout_desc const &desc,
  std::string_view name) -> detail::sync_sender_fn<compute_pipeline>
{
  return detail::make_sync_sender_fn<compute_pipeline>(
    [&ctx, strategy, glsl = std::string(glsl), desc, name = std::string(name)]() -> result<compute_pipeline> {
      VKEXEC_TRY_ASSIGN(owned, create_compute_resources(strategy, ctx, glsl, desc, name));
      return compute_pipeline::make(ctx, std::make_unique<pipeline_resources>(owned));
    });
}

}// namespace vkexec
