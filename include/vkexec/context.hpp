#ifndef VKEXEC_CONTEXT_HPP
#define VKEXEC_CONTEXT_HPP

#include <vkexec/pipeline_cache.hpp>

#include <VkBootstrap.h>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace vkexec {

class scheduler;
class window;

/// Handles borrowed from an embedder. vkexec never destroys these.
struct context_adopt_info
{
  VkInstance instance{ VK_NULL_HANDLE };
  VkPhysicalDevice physical_device{ VK_NULL_HANDLE };
  VkDevice device{ VK_NULL_HANDLE };
  VmaAllocator allocator{ VK_NULL_HANDLE };

  VkQueue compute_queue{ VK_NULL_HANDLE };
  std::uint32_t compute_queue_family{ 0 };

  VkQueue graphics_queue{ VK_NULL_HANDLE };
  std::uint32_t graphics_queue_family{ 0 };

  VkQueue present_queue{ VK_NULL_HANDLE };
  std::uint32_t present_queue_family{ 0 };
};

class context
{
public:
  /// Compute-only context (no window / swapchain).
  context();
  ~context();

  /// Wrap an existing Vulkan device/queues. Returns a context that does not destroy the
  /// instance, device, or an externally supplied VMA allocator. vkexec still owns its
  /// command pool and pipeline cache; create a VMA allocator when `allocator` is null.
  [[nodiscard]] static auto adopt(context_adopt_info const &info) -> std::unique_ptr<context>;

  context(context const &) = delete;
  auto operator=(context const &) -> context & = delete;
  context(context &&) noexcept = delete;
  auto operator=(context &&) noexcept -> context & = delete;

  [[nodiscard]] auto get_scheduler() noexcept -> scheduler;

  [[nodiscard]] auto instance() const noexcept -> VkInstance { return instance_.instance; }
  [[nodiscard]] auto physical_device() const noexcept -> VkPhysicalDevice { return physical_device_.physical_device; }
  [[nodiscard]] auto device() const noexcept -> VkDevice { return device_.device; }
  [[nodiscard]] auto vkb_device() noexcept -> vkb::Device & { return device_; }
  [[nodiscard]] auto vkb_device() const noexcept -> vkb::Device const & { return device_; }
  [[nodiscard]] auto compute_queue() const noexcept -> VkQueue { return compute_queue_; }
  [[nodiscard]] auto graphics_queue() const noexcept -> VkQueue { return graphics_queue_; }
  [[nodiscard]] auto present_queue() const noexcept -> VkQueue { return present_queue_; }
  [[nodiscard]] auto queue_family() const noexcept -> std::uint32_t { return queue_family_; }
  [[nodiscard]] auto graphics_queue_family() const noexcept -> std::uint32_t { return graphics_family_; }
  [[nodiscard]] auto present_queue_family() const noexcept -> std::uint32_t { return present_family_; }
  [[nodiscard]] auto command_pool() const noexcept -> VkCommandPool { return command_pool_; }
  [[nodiscard]] auto allocator() const noexcept -> VmaAllocator { return allocator_; }
  [[nodiscard]] auto get_pipeline_cache() noexcept -> pipeline_cache & { return *pipeline_cache_; }
  [[nodiscard]] auto presentation_enabled() const noexcept -> bool { return presentation_enabled_; }

  auto allocate_command_buffer() -> VkCommandBuffer;
  auto free_command_buffer(VkCommandBuffer cmd) -> void;

  auto submit_and_wait(VkCommandBuffer cmd) -> void;
  auto submit_async(VkCommandBuffer cmd, VkFence *out_fence = nullptr) -> VkSemaphore;

private:
  friend class pipeline_cache;
  friend class window;
  template<typename T> friend class buffer;

  struct instance_only_tag
  {
  };
  explicit context(instance_only_tag tag, std::vector<char const *> const &instance_extensions);
  explicit context(context_adopt_info const &info);
  auto complete_for_surface(VkSurfaceKHR surface) -> void;

  auto create_command_pool() -> void;
  auto create_allocator() -> void;
  auto fetch_queues(bool want_present) -> void;

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
  std::unique_ptr<pipeline_cache> pipeline_cache_;
  bool presentation_enabled_{ false };
  bool has_instance_{ false };
  bool has_device_{ false };
  bool owns_instance_{ false };
  bool owns_device_{ false };
  bool owns_allocator_{ false };
};

}// namespace vkexec

#endif// VKEXEC_CONTEXT_HPP
