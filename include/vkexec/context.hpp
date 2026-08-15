#pragma once

#include <vkexec/detail/pipeline_cache.hpp>

#include <VkBootstrap.h>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace vkexec {

class scheduler;
class window;

class context {
public:
  /// Compute-only context (no window / swapchain).
  context();
  ~context();

  context(const context &) = delete;
  context &operator=(const context &) = delete;
  context(context &&) noexcept = delete;
  context &operator=(context &&) noexcept = delete;

  [[nodiscard]] scheduler get_scheduler() noexcept;

  [[nodiscard]] VkInstance instance() const noexcept { return instance_.instance; }
  [[nodiscard]] VkPhysicalDevice physical_device() const noexcept { return physical_device_.physical_device; }
  [[nodiscard]] VkDevice device() const noexcept { return device_.device; }
  [[nodiscard]] vkb::Device &vkb_device() noexcept { return device_; }
  [[nodiscard]] vkb::Device const &vkb_device() const noexcept { return device_; }
  [[nodiscard]] VkQueue compute_queue() const noexcept { return compute_queue_; }
  [[nodiscard]] VkQueue graphics_queue() const noexcept { return graphics_queue_; }
  [[nodiscard]] VkQueue present_queue() const noexcept { return present_queue_; }
  [[nodiscard]] std::uint32_t queue_family() const noexcept { return queue_family_; }
  [[nodiscard]] std::uint32_t graphics_queue_family() const noexcept { return graphics_family_; }
  [[nodiscard]] std::uint32_t present_queue_family() const noexcept { return present_family_; }
  [[nodiscard]] VkCommandPool command_pool() const noexcept { return command_pool_; }
  [[nodiscard]] VmaAllocator allocator() const noexcept { return allocator_; }
  [[nodiscard]] PipelineCache &pipeline_cache() noexcept { return *pipeline_cache_; }
  [[nodiscard]] bool presentation_enabled() const noexcept { return presentation_enabled_; }

  VkCommandBuffer allocate_command_buffer();
  void free_command_buffer(VkCommandBuffer cmd);

  void submit_and_wait(VkCommandBuffer cmd);
  VkSemaphore submit_async(VkCommandBuffer cmd, VkFence *out_fence = nullptr);

private:
  friend class PipelineCache;
  friend class window;
  template<typename T>
  friend class buffer;

  struct instance_only_tag {};
  explicit context(instance_only_tag, std::vector<const char *> instance_extensions);
  void complete_for_surface(VkSurfaceKHR surface);

  void create_command_pool();
  void create_allocator();
  void fetch_queues(bool want_present);

  vkb::Instance instance_{};
  vkb::PhysicalDevice physical_device_{};
  vkb::Device device_{};
  VmaAllocator allocator_{ VK_NULL_HANDLE };
  VkQueue compute_queue_{ VK_NULL_HANDLE };
  VkQueue graphics_queue_{ VK_NULL_HANDLE };
  VkQueue present_queue_{ VK_NULL_HANDLE };
  std::uint32_t queue_family_{ 0 };
  std::uint32_t graphics_family_{ 0 };
  std::uint32_t present_family_{ 0 };
  VkCommandPool command_pool_{ VK_NULL_HANDLE };
  std::unique_ptr<PipelineCache> pipeline_cache_;
  bool presentation_enabled_{ false };
  bool has_instance_{ false };
  bool has_device_{ false };
};

} // namespace vkexec
