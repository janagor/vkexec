#ifndef VKEXEC_DETAIL_COMPUTE_CREATE_HPP
#define VKEXEC_DETAIL_COMPUTE_CREATE_HPP

#include <vkexec/context.hpp>
#include <vkexec/detail/compute_specialization.hpp>
#include <vkexec/detail/descriptor_backend.hpp>
#include <vkexec/detail/shader_module.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace vkexec::detail {

struct compute_create_info
{
  std::span<VkDescriptorSetLayoutBinding const> bindings;
  std::size_t push_bytes{ 0 };
  std::span<std::uint32_t const> specialization;
  std::array<std::uint32_t, 3> local_size{ k_default_local_size };
  bool create_binding_objects{ true };
};

template<descriptor_backend Backend>
auto destroy_compute_resources_with(context const &ctx, pipeline_resources &resources) noexcept -> void
{
  VkDevice device = ctx.device();
  if (resources.pipeline != VK_NULL_HANDLE) { vkDestroyPipeline(device, resources.pipeline, nullptr); }
  Backend::destroy_binding_objects(device, resources);
  if (resources.shader != VK_NULL_HANDLE) { vkDestroyShaderModule(device, resources.shader, nullptr); }
  resources = {};
}

template<descriptor_backend Backend>
[[nodiscard]] auto create_compute_resources_with(context &ctx,
  std::span<std::uint32_t const> spirv,
  compute_create_info const &info,
  std::string_view empty_message) -> result<pipeline_resources>
{
  if (spirv.empty()) { return fail(errc::invalid_argument, std::string(empty_message)); }

  pipeline_resources resources{};
  resources.binding_count = static_cast<std::uint32_t>(info.bindings.size());
  resources.push_bytes = info.push_bytes;
  resources.local_size = info.local_size;
  VkDevice device = ctx.device();

  auto shader = create_shader_module(device, spirv);
  if (!shader) { return fail(shader); }
  resources.shader = expected_take(shader);

  descriptor_layout_info const layout_info{ .bindings = info.bindings,
    .push_stages = VK_SHADER_STAGE_COMPUTE_BIT,
    .push_bytes = static_cast<std::uint32_t>(info.push_bytes),
    .create_set_layout = info.create_binding_objects,
    .create_pool = info.create_binding_objects };
  if (auto created = Backend::create_set_and_pipeline_layout(device, resources, layout_info); !created) {
    destroy_compute_resources_with<Backend>(ctx, resources);
    return fail(created);
  }

  auto const specialization = make_uint32_specialization(info.specialization);
  VkPipelineCreateFlags2CreateInfo flags_info{};
  flags_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO;
  flags_info.flags = Backend::pipeline_create_flags();

  // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.pNext = flags_info.flags == 0 ? nullptr : &flags_info;
  pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_info.stage.module = resources.shader;
  pipeline_info.stage.pName = "main";
  pipeline_info.stage.pSpecializationInfo = specialization.get();
  pipeline_info.layout = resources.pipeline_layout;
  if (VkResult const result =
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &resources.pipeline);
    result != VK_SUCCESS) {
    destroy_compute_resources_with<Backend>(ctx, resources);
    return fail(result, "vkCreateComputePipelines failed");
  }

  if (auto created = Backend::create_descriptor_pool(device, resources, layout_info); !created) {
    destroy_compute_resources_with<Backend>(ctx, resources);
    return fail(created);
  }
  return resources;
}

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_COMPUTE_CREATE_HPP
