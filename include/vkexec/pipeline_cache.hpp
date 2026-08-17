#ifndef VKEXEC_PIPELINE_CACHE_HPP
#define VKEXEC_PIPELINE_CACHE_HPP

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace vkexec::edsl {
struct ASTContext;
}

namespace vkexec {

class context;

enum class buffer_access : std::uint8_t { readonly, writeonly, readwrite };

inline constexpr std::uint32_t k_default_local_size_x = 64;
inline constexpr std::array<std::uint32_t, 3> k_default_local_size{ k_default_local_size_x, 1, 1 };

struct layout_desc
{
  /// Binding index is the position in this list (0, 1, ...).
  std::vector<buffer_access> bindings;
  std::size_t push_constant_size{ 0 };
  std::vector<std::uint32_t> specialization;
  std::array<std::uint32_t, 3> local_size{ k_default_local_size };
};

struct pipeline_resources
{
  VkShaderModule shader{ VK_NULL_HANDLE };
  VkDescriptorSetLayout set_layout{ VK_NULL_HANDLE };
  VkPipelineLayout pipeline_layout{ VK_NULL_HANDLE };
  VkPipeline pipeline{ VK_NULL_HANDLE };
  VkDescriptorPool descriptor_pool{ VK_NULL_HANDLE };
  std::uint32_t binding_count{ 0 };
  std::size_t push_bytes{ 0 };
  std::array<std::uint32_t, 3> local_size{ k_default_local_size };
};

class pipeline_cache
{
public:
  explicit pipeline_cache(context &ctx);
  ~pipeline_cache();

  pipeline_cache(pipeline_cache const &) = delete;
  auto operator=(pipeline_cache const &) -> pipeline_cache & = delete;
  pipeline_cache(pipeline_cache &&) = delete;
  auto operator=(pipeline_cache &&) -> pipeline_cache & = delete;

  auto get_or_compile(edsl::ASTContext const &ast, std::uint32_t work_count) -> pipeline_resources &;
  auto get_or_create_from_spirv(std::span<std::uint32_t const> spirv, layout_desc const &desc)
    -> pipeline_resources &;

private:
  context *ctx_;
  std::mutex mutex_;
  std::unordered_map<std::size_t, std::unique_ptr<pipeline_resources>> cache_;
};

}// namespace vkexec

#endif// VKEXEC_PIPELINE_CACHE_HPP
