#pragma once

#include <vkexec/detail/pipeline_cache.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace vkexec {

class scheduler;

class context {
public:
  context();
  ~context();

  context(const context &) = delete;
  context &operator=(const context &) = delete;
  context(context &&) noexcept = delete;
  context &operator=(context &&) noexcept = delete;

  [[nodiscard]] scheduler get_scheduler() noexcept;

  [[nodiscard]] VkInstance instance() const noexcept { return instance_; }
  [[nodiscard]] VkPhysicalDevice physical_device() const noexcept { return physical_; }
  [[nodiscard]] VkDevice device() const noexcept { return device_; }
  [[nodiscard]] VkQueue compute_queue() const noexcept { return queue_; }
  [[nodiscard]] std::uint32_t queue_family() const noexcept { return queue_family_; }
  [[nodiscard]] VkCommandPool command_pool() const noexcept { return command_pool_; }
  [[nodiscard]] PipelineCache &pipeline_cache() noexcept { return *pipeline_cache_; }

  VkCommandBuffer allocate_command_buffer();
  void free_command_buffer(VkCommandBuffer cmd);

  void submit_and_wait(VkCommandBuffer cmd);
  VkSemaphore submit_async(VkCommandBuffer cmd, VkFence *out_fence = nullptr);

private:
  friend class PipelineCache;
  template<typename T>
  friend class buffer;

  void create_instance();
  void pick_device();
  void create_device();
  void create_command_pool();

  VkInstance instance_{ VK_NULL_HANDLE };
  VkPhysicalDevice physical_{ VK_NULL_HANDLE };
  VkDevice device_{ VK_NULL_HANDLE };
  VkQueue queue_{ VK_NULL_HANDLE };
  std::uint32_t queue_family_{ 0 };
  VkCommandPool command_pool_{ VK_NULL_HANDLE };
  std::unique_ptr<PipelineCache> pipeline_cache_;
};

} // namespace vkexec
