#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/compute_specialization.hpp>
#include <vkexec/detail/shader_module.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/spirv_compile.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vkexec {
namespace {

  constexpr std::uint32_t k_descriptor_sets_per_pool = 64;

  auto create_set_layout(VkDevice device, std::uint32_t binding_count) -> result<VkDescriptorSetLayout>
  {
    std::vector<VkDescriptorSetLayoutBinding> bindings(binding_count);
    for (std::uint32_t index = 0; index < binding_count; ++index) {
      bindings.at(index).binding = index;
      bindings.at(index).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings.at(index).descriptorCount = 1;
      bindings.at(index).stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = binding_count;
    layout_info.pBindings = bindings.data();
    VkDescriptorSetLayout set_layout{ VK_NULL_HANDLE };
    VkResult const create_result = vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &set_layout);
    if (create_result != VK_SUCCESS) { return fail(create_result, "vkCreateDescriptorSetLayout failed"); }
    return set_layout;
  }

  auto create_pipeline_layout(VkDevice device, VkDescriptorSetLayout set_layout, std::size_t push_bytes)
    -> result<VkPipelineLayout>
  {
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = static_cast<std::uint32_t>(push_bytes);

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &set_layout;
    if (push_bytes > 0) {
      pipeline_layout_info.pushConstantRangeCount = 1;
      pipeline_layout_info.pPushConstantRanges = &push_range;
    }
    VkPipelineLayout layout{ VK_NULL_HANDLE };
    VkResult const create_result = vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &layout);
    if (create_result != VK_SUCCESS) { return fail(create_result, "vkCreatePipelineLayout failed"); }
    return layout;
  }

  auto create_vk_compute_pipeline(VkDevice device,
    VkShaderModule shader,
    VkPipelineLayout layout,
    VkSpecializationInfo const *specialization) -> result<VkPipeline>
  {
    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
    VkComputePipelineCreateInfo compute_info{};
    compute_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compute_info.stage.module = shader;
    compute_info.stage.pName = "main";
    compute_info.stage.pSpecializationInfo = specialization;
    compute_info.layout = layout;
    VkPipeline pipeline{ VK_NULL_HANDLE };
    VkResult const create_result =
      vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compute_info, nullptr, &pipeline);
    if (create_result != VK_SUCCESS) { return fail(create_result, "vkCreateComputePipelines failed"); }
    return pipeline;
  }

  auto create_descriptor_pool(VkDevice device, std::uint32_t binding_count) -> result<VkDescriptorPool>
  {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = std::max(1U, binding_count) * k_descriptor_sets_per_pool;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = k_descriptor_sets_per_pool;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VkDescriptorPool pool{ VK_NULL_HANDLE };
    VkResult const create_result = vkCreateDescriptorPool(device, &pool_info, nullptr, &pool);
    if (create_result != VK_SUCCESS) { return fail(create_result, "vkCreateDescriptorPool failed"); }
    return pool;
  }

}// namespace

auto destroy_compute_resources(context const &ctx, pipeline_resources &resources) noexcept -> void
{
  VkDevice device = ctx.device();
  if (resources.pipeline != VK_NULL_HANDLE) { vkDestroyPipeline(device, resources.pipeline, nullptr); }
  if (resources.pipeline_layout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device, resources.pipeline_layout, nullptr);
  }
  if (resources.descriptor_pool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device, resources.descriptor_pool, nullptr);
  }
  if (resources.set_layout != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device, resources.set_layout, nullptr); }
  if (resources.shader != VK_NULL_HANDLE) { vkDestroyShaderModule(device, resources.shader, nullptr); }
  resources = {};
}

auto create_compute_resources(context &ctx, std::span<std::uint32_t const> spirv, layout_desc const &desc)
  -> result<pipeline_resources>
{
  if (spirv.empty()) { return fail(errc::invalid_argument, "create_compute_resources requires non-empty SPIR-V"); }

  pipeline_resources owned{};
  owned.binding_count = static_cast<std::uint32_t>(desc.bindings.size());
  owned.push_bytes = desc.push_constant_size;
  owned.local_size = desc.local_size;

  auto const specialization = detail::make_uint32_specialization(desc.specialization);

  VkDevice device = ctx.device();

  auto shader_result = detail::create_shader_module(device, spirv);
  if (!shader_result) { return fail(shader_result); }
  owned.shader = expected_take(shader_result);

  auto set_layout_result = create_set_layout(device, owned.binding_count);
  if (!set_layout_result) {
    destroy_compute_resources(ctx, owned);
    return fail(set_layout_result);
  }
  owned.set_layout = expected_take(set_layout_result);

  auto pipeline_layout_result = create_pipeline_layout(device, owned.set_layout, desc.push_constant_size);
  if (!pipeline_layout_result) {
    destroy_compute_resources(ctx, owned);
    return fail(pipeline_layout_result);
  }
  owned.pipeline_layout = expected_take(pipeline_layout_result);

  auto pipeline_result = create_vk_compute_pipeline(device, owned.shader, owned.pipeline_layout, specialization.get());
  if (!pipeline_result) {
    destroy_compute_resources(ctx, owned);
    return fail(pipeline_result);
  }
  owned.pipeline = expected_take(pipeline_result);

  auto pool_result = create_descriptor_pool(device, owned.binding_count);
  if (!pool_result) {
    destroy_compute_resources(ctx, owned);
    return fail(pool_result);
  }
  owned.descriptor_pool = expected_take(pool_result);

  return owned;
}

auto create_compute_resources(context &ctx, std::string_view glsl, layout_desc const &desc, std::string_view name)
  -> result<pipeline_resources>
{
  if (glsl.empty()) { return fail(errc::invalid_argument, "create_compute_resources requires non-empty GLSL"); }
  VKEXEC_TRY_ASSIGN(spirv, compile_glsl_to_spirv(glsl, name, shader_kind::compute, ctx.api_version()));
  return create_compute_resources(ctx, spirv, desc);
}

auto allocate_compute_set(context const &ctx, pipeline_resources const &pipe) -> result<VkDescriptorSet>
{
  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = pipe.descriptor_pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &pipe.set_layout;
  VkDescriptorSet set{ VK_NULL_HANDLE };
  VkResult const allocate_result = vkAllocateDescriptorSets(ctx.device(), &dsai, &set);
  if (allocate_result != VK_SUCCESS) { return fail(allocate_result, "vkAllocateDescriptorSets failed"); }
  return set;
}

auto bind_storage(context &ctx, pipeline_resources const &pipe, std::span<storage_binding const> buffers)
  -> result<bound_compute>
{
  if (buffers.size() != pipe.binding_count) {
    return fail(errc::invalid_argument, "bind_storage buffer count must match layout_desc.bindings");
  }
  VKEXEC_TRY_ASSIGN(set, allocate_compute_set(ctx, pipe));
  write_storage_descriptors(ctx.device(), set, buffers);
  return bound_compute{ .pipe = &pipe, .set = set };
}

auto free_compute_set(context const &ctx, pipeline_resources const &pipe, VkDescriptorSet set) noexcept -> void
{
  if (set == VK_NULL_HANDLE || pipe.descriptor_pool == VK_NULL_HANDLE) { return; }
  vkFreeDescriptorSets(ctx.device(), pipe.descriptor_pool, 1, &set);
}

auto compute_pipeline::reset() noexcept -> void
{
  if (ctx_ != nullptr && resources_ != nullptr) { destroy_compute_resources(*ctx_, *resources_); }
  resources_.reset();
  ctx_ = nullptr;
}

auto compute_pipeline::create(context &ctx, std::span<std::uint32_t const> spirv, layout_desc const &desc)
  -> detail::sync_sender_fn<compute_pipeline>
{
  return detail::make_sync_sender_fn<compute_pipeline>([&ctx, spirv, desc]() -> result<compute_pipeline> {
    VKEXEC_TRY_ASSIGN(owned, create_compute_resources(ctx, spirv, desc));
    return compute_pipeline{ &ctx, std::make_unique<pipeline_resources>(owned) };
  });
}

auto compute_pipeline::create(context &ctx, std::string_view glsl, layout_desc const &desc, std::string_view name)
  -> detail::sync_sender_fn<compute_pipeline>
{
  return detail::make_sync_sender_fn<compute_pipeline>(
    [&ctx, glsl = std::string(glsl), desc, name = std::string(name)]() -> result<compute_pipeline> {
      // Capture by value: the sender may outlive the caller's string_views.
      VKEXEC_TRY_ASSIGN(owned, create_compute_resources(ctx, glsl, desc, name));
      return compute_pipeline{ &ctx, std::make_unique<pipeline_resources>(owned) };
    });
}

auto compute_pipeline::allocate_set_sender() const -> detail::sync_sender_fn<VkDescriptorSet>
{
  return detail::make_sync_sender_fn<VkDescriptorSet>([this]() -> result<VkDescriptorSet> { return allocate_set(); });
}

auto compute_pipeline::update_set_sender(VkDescriptorSet set, std::span<storage_binding const> buffers) const
  -> detail::sync_void_sender_fn
{
  std::vector<storage_binding> owned(buffers.begin(), buffers.end());
  return detail::make_sync_void_sender_fn(
    [this, set, owned = std::move(owned)]() -> status { return update_set(set, owned); });
}

auto compute_pipeline::allocate_set() const -> result<VkDescriptorSet>
{ return vkexec::allocate_compute_set(*ctx_, *resources_); }

auto compute_pipeline::update_set(VkDescriptorSet set, std::span<storage_binding const> buffers) const -> status
{
  if (buffers.size() != resources_->binding_count) {
    return fail(errc::invalid_argument, "update_set buffer count must match layout_desc.bindings");
  }
  write_storage_descriptors(ctx_->device(), set, buffers);
  return {};
}

auto bind_storage_sender(compute_pipeline const &pipe, std::span<storage_binding const> buffers)
  -> detail::sync_sender_fn<bound_compute_pipeline>
{
  std::vector<storage_binding> owned(buffers.begin(), buffers.end());
  return detail::make_sync_sender_fn<bound_compute_pipeline>(
    [&pipe, owned = std::move(owned)]() mutable -> result<bound_compute_pipeline> {
      VKEXEC_TRY_ASSIGN(set, pipe.allocate_set());
      VKEXEC_TRY(pipe.update_set(set, owned));
      return bound_compute_pipeline{ .pipe = &pipe, .set = set };
    });
}

auto compute_pass(compute_pipeline const &pipe, VkDescriptorSet set, std::uint32_t work_count)
  -> prebuilt_compute_pass_closure
{ return compute_pass(pipe.bind(set), pipe.groups_for(work_count)); }

}// namespace vkexec
