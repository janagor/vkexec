#include <vkexec/context.hpp>
#include <vkexec/detail/ast.hpp>
#include <vkexec/detail/glsl_emit.hpp>
#include <vkexec/detail/pipeline_cache.hpp>
#include <vkexec/detail/spirv.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace vkexec {
namespace {

void check(VkResult result, const char *what)
{
  if (result != VK_SUCCESS) { throw std::runtime_error(what); }
}

void destroy_resources(context &ctx, PipelineResources &r)
{
  VkDevice device = ctx.device();
  if (r.pipeline != VK_NULL_HANDLE) { vkDestroyPipeline(device, r.pipeline, nullptr); }
  if (r.pipeline_layout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, r.pipeline_layout, nullptr); }
  if (r.descriptor_pool != VK_NULL_HANDLE) { vkDestroyDescriptorPool(device, r.descriptor_pool, nullptr); }
  if (r.set_layout != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device, r.set_layout, nullptr); }
  if (r.shader != VK_NULL_HANDLE) { vkDestroyShaderModule(device, r.shader, nullptr); }
}

} // namespace

PipelineCache::PipelineCache(context &ctx) : ctx_(&ctx) {}

PipelineCache::~PipelineCache()
{
  for (auto &[_, res] : cache_) { destroy_resources(*ctx_, *res); }
  cache_.clear();
}

PipelineResources &PipelineCache::get_or_compile(const vlk::ASTContext &ast, std::uint32_t work_count)
{
  const std::size_t key = detail::hash_ast(ast) ^ (static_cast<std::size_t>(work_count) << 1U);
  {
    std::scoped_lock lock(mutex_);
    if (auto it = cache_.find(key); it != cache_.end()) { return *it->second; }
  }

  const std::string glsl = detail::emit_glsl(ast, work_count);
  const auto spirv = compile_glsl_to_spirv(glsl, "vkexec_bulk");

  auto res = std::make_unique<PipelineResources>();
  res->binding_count = static_cast<std::uint32_t>(ast.buffers.size());
  res->push_bytes = ast.push_bytes;

  VkShaderModuleCreateInfo smci{};
  smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smci.codeSize = spirv.size() * sizeof(std::uint32_t);
  smci.pCode = spirv.data();
  check(vkCreateShaderModule(ctx_->device(), &smci, nullptr, &res->shader), "vkCreateShaderModule failed");

  std::vector<VkDescriptorSetLayoutBinding> bindings(ast.buffers.size());
  for (std::size_t i = 0; i < ast.buffers.size(); ++i) {
    bindings[i].binding = static_cast<std::uint32_t>(ast.buffers[i].binding);
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = static_cast<std::uint32_t>(bindings.size());
  dslci.pBindings = bindings.data();
  check(vkCreateDescriptorSetLayout(ctx_->device(), &dslci, nullptr, &res->set_layout),
    "vkCreateDescriptorSetLayout failed");

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = static_cast<std::uint32_t>(ast.push_bytes);

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &res->set_layout;
  if (ast.push_bytes > 0) {
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
  }
  check(vkCreatePipelineLayout(ctx_->device(), &plci, nullptr, &res->pipeline_layout),
    "vkCreatePipelineLayout failed");

  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = res->shader;
  cpci.stage.pName = "main";
  cpci.layout = res->pipeline_layout;
  check(vkCreateComputePipelines(ctx_->device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &res->pipeline),
    "vkCreateComputePipelines failed");

  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = std::max(1u, res->binding_count) * 64u;

  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  dpci.maxSets = 64;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &pool_size;
  check(vkCreateDescriptorPool(ctx_->device(), &dpci, nullptr, &res->descriptor_pool),
    "vkCreateDescriptorPool failed");

  std::scoped_lock lock(mutex_);
  if (auto it = cache_.find(key); it != cache_.end()) {
    destroy_resources(*ctx_, *res);
    return *it->second;
  }
  auto [it, inserted] = cache_.emplace(key, std::move(res));
  (void)inserted;
  return *it->second;
}

} // namespace vkexec
