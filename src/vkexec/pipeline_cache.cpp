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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vkexec {
namespace {

constexpr std::uint32_t k_descriptor_sets_per_pool = 64;

auto check(VkResult result, const char *what) -> void
{
  if (result != VK_SUCCESS) { throw std::runtime_error(what); }
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
  if (resources.set_layout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device, resources.set_layout, nullptr);
  }
  if (resources.shader != VK_NULL_HANDLE) { vkDestroyShaderModule(device, resources.shader, nullptr); }
}

} // namespace

pipeline_cache::pipeline_cache(context &ctx) : ctx_(&ctx) {}

pipeline_cache::~pipeline_cache()
{
  for (auto &[key, resources] : cache_) {
    (void)key;
    destroy_resources(*ctx_, *resources);
  }
  cache_.clear();
}

auto pipeline_cache::get_or_compile(const edsl::ASTContext &ast, std::uint32_t work_count) -> pipeline_resources &
{
  const std::size_t key = edsl::hash_ast(ast) ^ (static_cast<std::size_t>(work_count) << 1U);
  {
    const std::scoped_lock lock(mutex_);
    if (auto cached = cache_.find(key); cached != cache_.end()) { return *cached->second; }
  }

  const std::string glsl = edsl::emit_glsl(ast, work_count);
  const auto spirv = edsl::compile_glsl_to_spirv(glsl, "vkexec_bulk");

  auto resources = std::make_unique<pipeline_resources>();
  resources->binding_count = static_cast<std::uint32_t>(ast.buffers.size());
  resources->push_bytes = ast.push_bytes;

  VkShaderModuleCreateInfo module_info{};
  module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  module_info.codeSize = spirv.size() * sizeof(std::uint32_t);
  module_info.pCode = spirv.data();
  check(vkCreateShaderModule(ctx_->device(), &module_info, nullptr, &resources->shader),
    "vkCreateShaderModule failed");

  std::vector<VkDescriptorSetLayoutBinding> bindings(ast.buffers.size());
  for (std::size_t index = 0; index < ast.buffers.size(); ++index) {
    bindings.at(index).binding = static_cast<std::uint32_t>(ast.buffers.at(index).binding);
    bindings.at(index).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings.at(index).descriptorCount = 1;
    bindings.at(index).stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
  layout_info.pBindings = bindings.data();
  check(vkCreateDescriptorSetLayout(ctx_->device(), &layout_info, nullptr, &resources->set_layout),
    "vkCreateDescriptorSetLayout failed");

  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.offset = 0;
  push_range.size = static_cast<std::uint32_t>(ast.push_bytes);

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &resources->set_layout;
  if (ast.push_bytes > 0) {
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
  }
  check(vkCreatePipelineLayout(ctx_->device(), &pipeline_layout_info, nullptr, &resources->pipeline_layout),
    "vkCreatePipelineLayout failed");

  // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
  VkComputePipelineCreateInfo compute_info{};
  compute_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  compute_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  compute_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  compute_info.stage.module = resources->shader;
  compute_info.stage.pName = "main";
  compute_info.layout = resources->pipeline_layout;
  check(vkCreateComputePipelines(ctx_->device(), VK_NULL_HANDLE, 1, &compute_info, nullptr, &resources->pipeline),
    "vkCreateComputePipelines failed");

  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = std::max(1U, resources->binding_count) * k_descriptor_sets_per_pool;

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = k_descriptor_sets_per_pool;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  check(vkCreateDescriptorPool(ctx_->device(), &pool_info, nullptr, &resources->descriptor_pool),
    "vkCreateDescriptorPool failed");

  const std::scoped_lock lock(mutex_);
  if (auto cached = cache_.find(key); cached != cache_.end()) {
    destroy_resources(*ctx_, *resources);
    return *cached->second;
  }
  auto [inserted_at, was_inserted] = cache_.emplace(key, std::move(resources));
  (void)was_inserted;
  return *inserted_at->second;
}

} // namespace vkexec
