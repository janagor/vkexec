#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>

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
auto destroy_heap_compute_resources(context const &ctx, pipeline_resources &resources) noexcept -> void
{ detail::destroy_compute_resources_with<detail::heap_descriptor_backend>(ctx, resources); }

auto create_heap_compute_resources(context &ctx, std::span<std::uint32_t const> spirv, heap_layout_desc const &desc)
  -> result<pipeline_resources>
{
  detail::compute_create_info const info{ .bindings = {},
    .push_bytes = 0,
    .specialization = desc.specialization,
    .local_size = desc.local_size,
    .create_binding_objects = false };
  return detail::create_compute_resources_with<detail::heap_descriptor_backend>(
    ctx, spirv, info, "create_heap_compute_resources requires non-empty SPIR-V");
}

auto create_heap_compute_resources(context &ctx,
  std::string_view glsl,
  heap_layout_desc const &desc,
  std::string_view name) -> result<pipeline_resources>
{
  if (glsl.empty()) { return fail(errc::invalid_argument, "create_heap_compute_resources requires non-empty GLSL"); }
  VKEXEC_TRY_ASSIGN(spirv, compile_glsl_to_spirv(glsl, name, shader_kind::compute, ctx.api_version()));
  return create_heap_compute_resources(ctx, spirv, desc);
}

auto heap_compute_pipeline::reset() noexcept -> void
{
  if (ctx_ != nullptr && resources_ != nullptr) { destroy_heap_compute_resources(*ctx_, *resources_); }
  resources_.reset();
  ctx_ = nullptr;
}

auto heap_compute_pipeline::create(context &ctx, std::span<std::uint32_t const> spirv, heap_layout_desc const &desc)
  -> detail::sync_sender_fn<heap_compute_pipeline>
{
  return detail::make_sync_sender_fn<heap_compute_pipeline>([&ctx, spirv, desc]() -> result<heap_compute_pipeline> {
    VKEXEC_TRY_ASSIGN(owned, create_heap_compute_resources(ctx, spirv, desc));
    return make(ctx, std::make_unique<pipeline_resources>(owned));
  });
}

auto heap_compute_pipeline::create(context &ctx,
  std::string_view glsl,
  heap_layout_desc const &desc,
  std::string_view name) -> detail::sync_sender_fn<heap_compute_pipeline>
{
  return detail::make_sync_sender_fn<heap_compute_pipeline>(
    [&ctx, glsl = std::string(glsl), desc, name = std::string(name)]() -> result<heap_compute_pipeline> {
      VKEXEC_TRY_ASSIGN(owned, create_heap_compute_resources(ctx, glsl, desc, name));
      return make(ctx, std::make_unique<pipeline_resources>(owned));
    });
}

}// namespace vkexec
