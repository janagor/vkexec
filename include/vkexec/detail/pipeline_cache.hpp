#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vlk {
struct ASTContext;
}

namespace vkexec {

class context;

struct PipelineResources {
  VkShaderModule shader{ VK_NULL_HANDLE };
  VkDescriptorSetLayout set_layout{ VK_NULL_HANDLE };
  VkPipelineLayout pipeline_layout{ VK_NULL_HANDLE };
  VkPipeline pipeline{ VK_NULL_HANDLE };
  VkDescriptorPool descriptor_pool{ VK_NULL_HANDLE };
  std::uint32_t binding_count{ 0 };
  std::size_t push_bytes{ 0 };
};

class PipelineCache {
public:
  explicit PipelineCache(context &ctx);
  ~PipelineCache();

  PipelineCache(const PipelineCache &) = delete;
  PipelineCache &operator=(const PipelineCache &) = delete;

  PipelineResources &get_or_compile(const vlk::ASTContext &ast, std::uint32_t work_count);

private:
  context *ctx_;
  std::mutex mutex_;
  std::unordered_map<std::size_t, std::unique_ptr<PipelineResources>> cache_;
};

} // namespace vkexec
