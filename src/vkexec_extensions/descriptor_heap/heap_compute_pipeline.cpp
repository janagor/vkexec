#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>

#include "detail/strategy.hpp"

#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/compute_create.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec_extensions/descriptor_heap/strategy.hpp>

#include <cstdint>
#include <memory>
#include <span>

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

auto create_compute_pipeline(descriptor_heap_t strategy,
  context &ctx,
  std::span<std::uint32_t const> spirv,
  heap_layout_desc const &desc) -> sender<compute_pipeline>
{
  return make_sender<compute_pipeline>([&ctx, strategy, spirv, desc]() -> result<compute_pipeline> {
    VKEXEC_TRY_ASSIGN(owned, create_compute_resources(strategy, ctx, spirv, desc));
    return compute_pipeline::make(ctx, std::make_unique<pipeline_resources>(owned));
  });
}

}// namespace vkexec
