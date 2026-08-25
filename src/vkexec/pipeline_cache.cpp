#include "pipeline_cache.hpp"

#include <vkexec/context.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec_edsl/spirv.hpp>

#include "vkexec_edsl/ast.hpp"
#include "vkexec_edsl/glsl_emit.hpp"

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace vkexec {
namespace {

  constexpr std::uint32_t k_descriptor_sets_per_pool = 64;
  constexpr std::size_t k_spirv_cache_tag = 0xC0DE51B5ULL;
  constexpr std::size_t k_hash_golden_ratio = 0x9e3779b97f4a7c15ULL;
  constexpr std::size_t k_hash_shift_left = 6U;
  constexpr std::size_t k_hash_shift_right = 2U;

  auto hash_combine(std::size_t seed, std::size_t value) -> std::size_t
  { return seed ^ (value + k_hash_golden_ratio + (seed << k_hash_shift_left) + (seed >> k_hash_shift_right)); }

  auto hash_spirv_layout(std::span<std::uint32_t const> spirv, layout_desc const &desc) -> std::size_t
  {
    std::size_t hash = k_spirv_cache_tag;
    hash = hash_combine(hash, spirv.size());
    // cppcheck-suppress useStlAlgorithm
    for (std::uint32_t const word : spirv) { hash = hash_combine(hash, word); }
    hash = hash_combine(hash, desc.bindings.size());
    // cppcheck-suppress useStlAlgorithm
    for (buffer_access const access : desc.bindings) { hash = hash_combine(hash, static_cast<std::size_t>(access)); }
    hash = hash_combine(hash, desc.push_constant_size);
    hash = hash_combine(hash, desc.specialization.size());
    // cppcheck-suppress useStlAlgorithm
    for (std::uint32_t const value : desc.specialization) { hash = hash_combine(hash, value); }
    hash = hash_combine(hash, desc.local_size.at(0));
    hash = hash_combine(hash, desc.local_size.at(1));
    hash = hash_combine(hash, desc.local_size.at(2));
    hash = hash_combine(hash, desc.descriptor_heap ? 1U : 0U);
    return hash;
  }

  auto destroy_resources(context const &ctx, pipeline_resources &resources) -> void
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
  }

  auto create_shader_module(VkDevice device, std::span<std::uint32_t const> spirv) -> result<VkShaderModule>
  {
    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = spirv.size_bytes();
    module_info.pCode = spirv.data();
    VkShaderModule shader{ VK_NULL_HANDLE };
    VkResult const create_result = vkCreateShaderModule(device, &module_info, nullptr, &shader);
    if (create_result != VK_SUCCESS) {
      return std::unexpected(make_vk_error(create_result, "vkCreateShaderModule failed"));
    }
    return shader;
  }

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
    if (create_result != VK_SUCCESS) {
      return std::unexpected(make_vk_error(create_result, "vkCreateDescriptorSetLayout failed"));
    }
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
    if (create_result != VK_SUCCESS) {
      return std::unexpected(make_vk_error(create_result, "vkCreatePipelineLayout failed"));
    }
    return layout;
  }

  auto create_compute_pipeline(VkDevice device,
    VkShaderModule shader,
    VkPipelineLayout layout,
    VkSpecializationInfo const *specialization,
    bool descriptor_heap) -> result<VkPipeline>
  {
    VkPipelineCreateFlags2CreateInfo flags2{};
    flags2.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO;
    flags2.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
    VkComputePipelineCreateInfo compute_info{};
    compute_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_info.pNext = descriptor_heap ? &flags2 : nullptr;
    compute_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compute_info.stage.module = shader;
    compute_info.stage.pName = "main";
    compute_info.stage.pSpecializationInfo = specialization;
    compute_info.layout = layout;
    VkPipeline pipeline{ VK_NULL_HANDLE };
    VkResult const create_result =
      vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compute_info, nullptr, &pipeline);
    if (create_result != VK_SUCCESS) {
      return std::unexpected(make_vk_error(create_result, "vkCreateComputePipelines failed"));
    }
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
    if (create_result != VK_SUCCESS) {
      return std::unexpected(make_vk_error(create_result, "vkCreateDescriptorPool failed"));
    }
    return pool;
  }

}// namespace

pipeline_cache::pipeline_cache(context &ctx) : ctx_(&ctx) {}

pipeline_cache::~pipeline_cache()
{
  for (auto &[key, resources] : cache_) {
    (void)key;
    destroy_resources(*ctx_, *resources);
  }
  cache_.clear();
}

auto pipeline_cache::get_or_compile(edsl::ASTContext const &ast, std::uint32_t work_count)
  -> result<std::reference_wrapper<pipeline_resources>>
{
  std::size_t const key = edsl::hash_ast(ast) ^ (static_cast<std::size_t>(work_count) << 1U);
  {
    std::scoped_lock const lock(mutex_);
    if (auto cached = cache_.find(key); cached != cache_.end()) { return std::ref(*cached->second); }
  }

  std::string const glsl = edsl::emit_glsl(ast, work_count);
  auto const spirv = edsl::compile_glsl_to_spirv(glsl, "vkexec_bulk", edsl::shader_kind::compute, ctx_->api_version());
  if (!spirv) { return std::unexpected(spirv.error()); }

  auto resources = std::make_unique<pipeline_resources>();
  resources->binding_count = static_cast<std::uint32_t>(ast.buffers.size());
  resources->push_bytes = ast.push_bytes;
  resources->local_size = { static_cast<std::uint32_t>(ast.local_size_x), 1, 1 };

  VkDevice device = ctx_->device();

  auto shader_result = create_shader_module(device, *spirv);
  if (!shader_result) { return propagate(shader_result); }
  resources->shader = *shader_result;

  auto set_layout_result = create_set_layout(device, resources->binding_count);
  if (!set_layout_result) {
    destroy_resources(*ctx_, *resources);
    return propagate(set_layout_result);
  }
  resources->set_layout = *set_layout_result;

  auto pipeline_layout_result = create_pipeline_layout(device, resources->set_layout, ast.push_bytes);
  if (!pipeline_layout_result) {
    destroy_resources(*ctx_, *resources);
    return propagate(pipeline_layout_result);
  }
  resources->pipeline_layout = *pipeline_layout_result;

  auto pipeline_result =
    create_compute_pipeline(device, resources->shader, resources->pipeline_layout, nullptr, false);
  if (!pipeline_result) {
    destroy_resources(*ctx_, *resources);
    return propagate(pipeline_result);
  }
  resources->pipeline = *pipeline_result;

  auto pool_result = create_descriptor_pool(device, resources->binding_count);
  if (!pool_result) {
    destroy_resources(*ctx_, *resources);
    return propagate(pool_result);
  }
  resources->descriptor_pool = *pool_result;

  std::scoped_lock const lock(mutex_);
  if (auto cached = cache_.find(key); cached != cache_.end()) {
    destroy_resources(*ctx_, *resources);
    return std::ref(*cached->second);
  }
  auto [inserted_at, was_inserted] = cache_.emplace(key, std::move(resources));
  (void)was_inserted;
  return std::ref(*inserted_at->second);
}

auto pipeline_cache::get_or_create_from_spirv(std::span<std::uint32_t const> spirv, layout_desc const &desc)
  -> result<std::reference_wrapper<pipeline_resources>>
{
  if (spirv.empty()) {
    return std::unexpected(make_error(errc::invalid_argument, "compute_pipeline::create requires non-empty SPIR-V"));
  }
  std::size_t const key = hash_spirv_layout(spirv, desc);
  {
    std::scoped_lock const lock(mutex_);
    if (auto cached = cache_.find(key); cached != cache_.end()) { return std::ref(*cached->second); }
  }

  if (desc.descriptor_heap) {
    if (!desc.bindings.empty()) {
      return std::unexpected(
        make_error(errc::invalid_argument, "descriptor_heap pipelines must not declare descriptor-set bindings"));
    }
    if (desc.push_constant_size != 0) {
      return std::unexpected(
        make_error(errc::invalid_argument, "descriptor_heap pipelines use push data, not push constants"));
    }
  }

  auto resources = std::make_unique<pipeline_resources>();
  resources->binding_count = static_cast<std::uint32_t>(desc.bindings.size());
  resources->push_bytes = desc.push_constant_size;
  resources->local_size = desc.local_size;

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

  VkDevice device = ctx_->device();

  auto shader_result = create_shader_module(device, spirv);
  if (!shader_result) { return propagate(shader_result); }
  resources->shader = *shader_result;

  if (desc.descriptor_heap) {
    auto pipeline_result = create_compute_pipeline(device, resources->shader, VK_NULL_HANDLE, spec_ptr, true);
    if (!pipeline_result) {
      destroy_resources(*ctx_, *resources);
      return propagate(pipeline_result);
    }
    resources->pipeline = *pipeline_result;
  } else {
    auto set_layout_result = create_set_layout(device, resources->binding_count);
    if (!set_layout_result) {
      destroy_resources(*ctx_, *resources);
      return propagate(set_layout_result);
    }
    resources->set_layout = *set_layout_result;

    auto pipeline_layout_result = create_pipeline_layout(device, resources->set_layout, desc.push_constant_size);
    if (!pipeline_layout_result) {
      destroy_resources(*ctx_, *resources);
      return propagate(pipeline_layout_result);
    }
    resources->pipeline_layout = *pipeline_layout_result;

    auto pipeline_result =
      create_compute_pipeline(device, resources->shader, resources->pipeline_layout, spec_ptr, false);
    if (!pipeline_result) {
      destroy_resources(*ctx_, *resources);
      return propagate(pipeline_result);
    }
    resources->pipeline = *pipeline_result;

    auto pool_result = create_descriptor_pool(device, resources->binding_count);
    if (!pool_result) {
      destroy_resources(*ctx_, *resources);
      return propagate(pool_result);
    }
    resources->descriptor_pool = *pool_result;
  }

  std::scoped_lock const lock(mutex_);
  if (auto cached = cache_.find(key); cached != cache_.end()) {
    destroy_resources(*ctx_, *resources);
    return std::ref(*cached->second);
  }
  auto [inserted_at, was_inserted] = cache_.emplace(key, std::move(resources));
  (void)was_inserted;
  return std::ref(*inserted_at->second);
}

}// namespace vkexec
