#ifndef VKEXEC_PIPELINE_CACHE_HPP
#define VKEXEC_PIPELINE_CACHE_HPP

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace vkexec::edsl {
struct ASTContext;
}

namespace vkexec {

class context;

struct pipeline_resources
{
  VkShaderModule shader{ VK_NULL_HANDLE };
  VkDescriptorSetLayout set_layout{ VK_NULL_HANDLE };
  VkPipelineLayout pipeline_layout{ VK_NULL_HANDLE };
  VkPipeline pipeline{ VK_NULL_HANDLE };
  VkDescriptorPool descriptor_pool{ VK_NULL_HANDLE };
  std::uint32_t binding_count{ 0 };
  std::size_t push_bytes{ 0 };
};

class pipeline_cache
{
public:
  explicit pipeline_cache(context &ctx);
  ~pipeline_cache();

  pipeline_cache(const pipeline_cache &) = delete;
  auto operator=(const pipeline_cache &) -> pipeline_cache & = delete;
  pipeline_cache(pipeline_cache &&) = delete;
  auto operator=(pipeline_cache &&) -> pipeline_cache & = delete;

  auto get_or_compile(const edsl::ASTContext &ast, std::uint32_t work_count) -> pipeline_resources &;

private:
  context *ctx_;
  std::mutex mutex_;
  std::unordered_map<std::size_t, std::unique_ptr<pipeline_resources>> cache_;
};

}// namespace vkexec

#endif// VKEXEC_PIPELINE_CACHE_HPP
