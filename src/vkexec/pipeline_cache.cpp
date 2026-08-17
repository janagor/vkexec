#include <vkexec/context.hpp>
#include <vkexec/pipeline_cache.hpp>
#include <vkexec_edsl/ast.hpp>
#include <vkexec_edsl/glsl_emit.hpp>
#include <vkexec_edsl/spirv.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
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

  auto check(VkResult result, char const *what) -> void
  {
    if (result != VK_SUCCESS) { throw std::runtime_error(what); }
  }

  auto hash_combine(std::size_t seed, std::size_t value) -> std::size_t
  {
    return seed ^ (value + k_hash_golden_ratio + (seed << k_hash_shift_left) + (seed >> k_hash_shift_right));
  }

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

  auto create_shader_module(VkDevice device, std::span<std::uint32_t const> spirv) -> VkShaderModule
  {
    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = spirv.size_bytes();
    module_info.pCode = spirv.data();
    VkShaderModule shader{ VK_NULL_HANDLE };
    check(vkCreateShaderModule(device, &module_info, nullptr, &shader), "vkCreateShaderModule failed");
    return shader;
  }

  auto create_set_layout(VkDevice device, std::uint32_t binding_count) -> VkDescriptorSetLayout
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
    check(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &set_layout),
      "vkCreateDescriptorSetLayout failed");
    return set_layout;
  }

  auto create_pipeline_layout(VkDevice device, VkDescriptorSetLayout set_layout, std::size_t push_bytes)
    -> VkPipelineLayout
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
    check(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &layout), "vkCreatePipelineLayout failed");
    return layout;
  }

  auto create_compute_pipeline(VkDevice device,
    VkShaderModule shader,
    VkPipelineLayout layout,
    VkSpecializationInfo const *specialization) -> VkPipeline
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
    check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compute_info, nullptr, &pipeline),
      "vkCreateComputePipelines failed");
    return pipeline;
  }

  auto create_descriptor_pool(VkDevice device, std::uint32_t binding_count) -> VkDescriptorPool
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
    check(vkCreateDescriptorPool(device, &pool_info, nullptr, &pool), "vkCreateDescriptorPool failed");
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

auto pipeline_cache::get_or_compile(edsl::ASTContext const &ast, std::uint32_t work_count) -> pipeline_resources &
{
  std::size_t const key = edsl::hash_ast(ast) ^ (static_cast<std::size_t>(work_count) << 1U);
  {
    std::scoped_lock const lock(mutex_);
    if (auto cached = cache_.find(key); cached != cache_.end()) { return *cached->second; }
  }

  std::string const glsl = edsl::emit_glsl(ast, work_count);
  auto const spirv = edsl::compile_glsl_to_spirv(glsl, "vkexec_bulk");

  auto resources = std::make_unique<pipeline_resources>();
  resources->binding_count = static_cast<std::uint32_t>(ast.buffers.size());
  resources->push_bytes = ast.push_bytes;
  resources->local_size = { static_cast<std::uint32_t>(ast.local_size_x), 1, 1 };

  VkDevice device = ctx_->device();
  resources->shader = create_shader_module(device, spirv);
  resources->set_layout = create_set_layout(device, resources->binding_count);
  resources->pipeline_layout = create_pipeline_layout(device, resources->set_layout, ast.push_bytes);
  resources->pipeline = create_compute_pipeline(device, resources->shader, resources->pipeline_layout, nullptr);
  resources->descriptor_pool = create_descriptor_pool(device, resources->binding_count);

  std::scoped_lock const lock(mutex_);
  if (auto cached = cache_.find(key); cached != cache_.end()) {
    destroy_resources(*ctx_, *resources);
    return *cached->second;
  }
  auto [inserted_at, was_inserted] = cache_.emplace(key, std::move(resources));
  (void)was_inserted;
  return *inserted_at->second;
}

auto pipeline_cache::get_or_create_from_spirv(std::span<std::uint32_t const> spirv, layout_desc const &desc)
  -> pipeline_resources &
{
  if (spirv.empty()) { throw std::invalid_argument("compute_pipeline::from_spirv requires non-empty SPIR-V"); }
  std::size_t const key = hash_spirv_layout(spirv, desc);
  {
    std::scoped_lock const lock(mutex_);
    if (auto cached = cache_.find(key); cached != cache_.end()) { return *cached->second; }
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
  resources->shader = create_shader_module(device, spirv);
  resources->set_layout = create_set_layout(device, resources->binding_count);
  resources->pipeline_layout = create_pipeline_layout(device, resources->set_layout, desc.push_constant_size);
  resources->pipeline = create_compute_pipeline(device, resources->shader, resources->pipeline_layout, spec_ptr);
  resources->descriptor_pool = create_descriptor_pool(device, resources->binding_count);

  std::scoped_lock const lock(mutex_);
  if (auto cached = cache_.find(key); cached != cache_.end()) {
    destroy_resources(*ctx_, *resources);
    return *cached->second;
  }
  auto [inserted_at, was_inserted] = cache_.emplace(key, std::move(resources));
  (void)was_inserted;
  return *inserted_at->second;
}

}// namespace vkexec
