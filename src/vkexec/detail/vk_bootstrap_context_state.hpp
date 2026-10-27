#ifndef VKEXEC_DETAIL_VK_BOOTSTRAP_CONTEXT_STATE_HPP
#define VKEXEC_DETAIL_VK_BOOTSTRAP_CONTEXT_STATE_HPP

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/object_synchronization.hpp>
#include <vkexec/detail/synchronization.hpp>
#include <vkexec/device_capabilities.hpp>
#include <vkexec/device_procs.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/result.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include "completion_waiter.hpp"
#include "host_agent.hpp"

#include <VkBootstrap.h>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vkexec {

struct detail::context_state
{
  vulkan_requirements creation_requirements{};
  device_capabilities capabilities{};
  device_procs procs{};
  std::mutex extension_procs_mutex;
  std::map<std::string, std::shared_ptr<void>, std::less<>> extension_procs;
  detail::synchronization_backend synchronization{ detail::synchronization_backend::legacy };
  vkb::Instance instance{};
  vkb::PhysicalDevice physical_device{};
  vkb::Device device{};
  VkQueue compute_queue{ VK_NULL_HANDLE };
  VkQueue graphics_queue{ VK_NULL_HANDLE };
  VkQueue present_queue{ VK_NULL_HANDLE };
  std::uint32_t queue_family{ 0 };
  std::uint32_t graphics_family{ 0 };
  std::uint32_t present_family{ 0 };
  std::unique_ptr<detail::completion_waiter> completion_waiter;
  std::unique_ptr<detail::host_agent> host_agent;
  mutable detail::queue_synchronization_registry queue_synchronization;
  mutable detail::descriptor_pool_synchronization_registry descriptor_pool_synchronization;
  // A checked-out pool has exactly one host owner; only returned pools appear here.
  std::mutex command_pool_cache_mutex;
  std::vector<std::pair<std::uint32_t, VkCommandPool>> available_command_pools;
  std::mutex command_buffer_mutex;
  std::unordered_map<VkCommandBuffer, std::pair<std::uint32_t, VkCommandPool>> command_buffer_pools;
  bool presentation_enabled{ false };
  bool has_instance{ false };
  bool has_device{ false };
  bool owns_instance{ false };
  bool owns_device{ false };
  context_state *next_retired{ nullptr };
  std::shared_ptr<std::promise<void>> retirement_done{ std::make_shared<std::promise<void>>() };
  std::shared_future<void> retirement{ retirement_done->get_future().share() };

  ~context_state();
  context_state() = default;
  context_state(context_state const &) = delete;
  auto operator=(context_state const &) -> context_state & = delete;
  context_state(context_state &&) = delete;
  auto operator=(context_state &&) -> context_state & = delete;

  auto register_known_queues() -> status
  {
#if VKEXEC_HAS_EXCEPTIONS
    try {
#endif
      auto register_queue = [this](VkQueue queue) -> void {
        if (queue != VK_NULL_HANDLE) { (void)queue_synchronization.state(queue); }
      };
      register_queue(compute_queue);
      register_queue(graphics_queue);
      register_queue(present_queue);
#if VKEXEC_HAS_EXCEPTIONS
    } catch (...) {
      return fail(unexpected_exception_error());
    }
#endif
    return {};
  }
};

} // namespace vkexec

#endif// VKEXEC_DETAIL_VK_BOOTSTRAP_CONTEXT_STATE_HPP
