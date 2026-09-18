#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>

#include <vkexec/context.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/spirv_compile.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace vkexec {
namespace {

  auto create_shader_module(VkDevice device, std::span<std::uint32_t const> spirv) -> result<VkShaderModule>
  {
    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = spirv.size_bytes();
    module_info.pCode = spirv.data();
    VkShaderModule shader{ VK_NULL_HANDLE };
    VkResult const create_result = vkCreateShaderModule(device, &module_info, nullptr, &shader);
    if (create_result != VK_SUCCESS) { return fail(create_result, "vkCreateShaderModule failed"); }
    return shader;
  }

  auto create_heap_vk_pipeline(VkDevice device, VkShaderModule shader, VkSpecializationInfo const *specialization)
    -> result<VkPipeline>
  {
    VkPipelineCreateFlags2CreateInfo flags2{};
    flags2.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO;
    flags2.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
    VkComputePipelineCreateInfo compute_info{};
    compute_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_info.pNext = &flags2;
    compute_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compute_info.stage.module = shader;
    compute_info.stage.pName = "main";
    compute_info.stage.pSpecializationInfo = specialization;
    compute_info.layout = VK_NULL_HANDLE;
    VkPipeline pipeline{ VK_NULL_HANDLE };
    VkResult const create_result =
      vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compute_info, nullptr, &pipeline);
    if (create_result != VK_SUCCESS) { return fail(create_result, "vkCreateComputePipelines failed"); }
    return pipeline;
  }

}// namespace

auto destroy_heap_compute_resources(context const &ctx, pipeline_resources &resources) noexcept -> void
{
  VkDevice device = ctx.device();
  if (resources.pipeline != VK_NULL_HANDLE) { vkDestroyPipeline(device, resources.pipeline, nullptr); }
  if (resources.shader != VK_NULL_HANDLE) { vkDestroyShaderModule(device, resources.shader, nullptr); }
  resources = {};
}

auto create_heap_compute_resources(context &ctx, std::span<std::uint32_t const> spirv, heap_layout_desc const &desc)
  -> result<pipeline_resources>
{
  if (spirv.empty()) { return fail(errc::invalid_argument, "create_heap_compute_resources requires non-empty SPIR-V"); }

  pipeline_resources resources{};
  resources.local_size = desc.local_size;

  // Same constantID = index convention as classic compute_pipeline.
  std::vector<VkSpecializationMapEntry> spec_entries(desc.specialization.size());
  for (std::size_t index = 0; index < desc.specialization.size(); ++index) {
    spec_entries.at(index).constantID = static_cast<std::uint32_t>(index);
    spec_entries.at(index).offset = static_cast<std::uint32_t>(index * sizeof(std::uint32_t));
    spec_entries.at(index).size = sizeof(std::uint32_t);
  }
  VkSpecializationInfo spec_info{};
  VkSpecializationInfo const *spec_ptr = nullptr;
  if (!desc.specialization.empty()) {
    spec_info.mapEntryCount = static_cast<std::uint32_t>(spec_entries.size());
    spec_info.pMapEntries = spec_entries.data();
    spec_info.dataSize = desc.specialization.size() * sizeof(std::uint32_t);
    spec_info.pData = desc.specialization.data();
    spec_ptr = &spec_info;
  }

  VkDevice device = ctx.device();
  auto shader_result = create_shader_module(device, spirv);
  if (!shader_result) { return fail(shader_result); }
  resources.shader = expected_take(shader_result);

  auto pipeline_result = create_heap_vk_pipeline(device, resources.shader, spec_ptr);
  if (!pipeline_result) {
    destroy_heap_compute_resources(ctx, resources);
    return fail(pipeline_result);
  }
  resources.pipeline = expected_take(pipeline_result);

  return resources;
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
