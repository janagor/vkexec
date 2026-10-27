#include <VkBootstrap.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/object_synchronization.hpp>
#include <vkexec/detail/submission.hpp>
#include <vkexec/detail/worker_callbacks.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>
#include <vulkan/vulkan_core.h>

#include "detail/completion_waiter.hpp"
#include "detail/host_agent.hpp"
#include "detail/retirement_wait.hpp"
#include "detail/test_vulkan_dispatch.hpp"
#include "detail/vk_bootstrap_context_state.hpp"
#include "vkexec/device_capabilities.hpp"

namespace vkexec {

context::~context() = default;

detail::context_state::~context_state()
{
  // Agents first so outstanding completions finish before tearing down Vulkan objects.
  host_agent.reset();
  completion_waiter.reset();

  if (device.device != VK_NULL_HANDLE) {
    if (owns_device) {
      bool retired = false;
      for (std::uint32_t attempt = 0; attempt < detail::k_retirement_wait_attempts; ++attempt) {
        VkResult const waited = vkDeviceWaitIdle(device.device);
        if (waited == VK_SUCCESS || waited == VK_ERROR_DEVICE_LOST) {
          retired = true;
          break;
        }
      }
      if (!retired) { std::terminate(); }
    }
    for (auto const &[cmd, family_pool] : command_buffer_pools) {
      (void)cmd;
      vkDestroyCommandPool(device.device, family_pool.second, nullptr);
    }
    for (auto const &[family, pool] : available_command_pools) {
      (void)family;
      vkDestroyCommandPool(device.device, pool, nullptr);
    }
    if (owns_device) {
      vkb::destroy_device(device);
      has_device = false;
    }
  }

  if (owns_instance && has_instance) {
    vkb::destroy_instance(instance);
    has_instance = false;
  }
}

auto context::lock_queue(VkQueue queue) const -> queue_guard
{ return queue_guard{ impl_->queue_synchronization.state(queue) }; }

auto detail::descriptor_pool_access::lock(context const &ctx, VkDescriptorPool pool) -> std::unique_lock<std::mutex>
{ return std::unique_lock{ *ctx.impl_->descriptor_pool_synchronization.state(pool) }; }

auto detail::descriptor_pool_access::register_owned(context const &ctx, VkDescriptorPool pool) -> status
{
  if (pool == VK_NULL_HANDLE) { return {}; }
#if VKEXEC_HAS_EXCEPTIONS
  try {
#endif
    (void)ctx.impl_->descriptor_pool_synchronization.state(pool);
#if VKEXEC_HAS_EXCEPTIONS
  } catch (...) {
    return fail(unexpected_exception_error());
  }
#endif
  return {};
}

auto context::acquire_command_pool(std::uint32_t family) -> result<VkCommandPool>
{
  {
    std::scoped_lock const lock(impl_->command_pool_cache_mutex);
    auto found = std::ranges::find_if(
      impl_->available_command_pools, [family](auto const &entry) -> bool { return entry.first == family; });
    if (found != impl_->available_command_pools.end()) {
      VkCommandPool pool = found->second;
      impl_->available_command_pools.erase(found);
      return pool;
    }
  }
  VkCommandPoolCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  info.queueFamilyIndex = family;
  info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkCommandPool pool{ VK_NULL_HANDLE };
  if (VkResult const result = vkCreateCommandPool(device(), &info, nullptr, &pool); result != VK_SUCCESS) {
    return fail(result, "vkCreateCommandPool failed");
  }
  return pool;
}

auto context::release_command_pool(std::uint32_t family, VkCommandPool pool) noexcept -> void
{
  if (pool == VK_NULL_HANDLE) { return; }
  if (vkResetCommandPool(device(), pool, 0) != VK_SUCCESS) {
    vkDestroyCommandPool(device(), pool, nullptr);
    return;
  }
#if VKEXEC_HAS_EXCEPTIONS
  try {
#endif
    std::scoped_lock const lock(impl_->command_pool_cache_mutex);
    impl_->available_command_pools.emplace_back(family, pool);
#if VKEXEC_HAS_EXCEPTIONS
  } catch (...) {
    vkDestroyCommandPool(device(), pool, nullptr);
  }
#endif
}

auto context::host_agent_thread_id() -> std::thread::id
{
  if (!impl_->host_agent) { impl_->host_agent = std::make_unique<detail::host_agent>(); }
  return impl_->host_agent->thread_id();
}

auto context::do_enqueue_fence_wait(VkSemaphore semaphore,
  VkFence fence,
  detail::stop_fn stop_requested,
  detail::done_fn on_done) -> status
{
  if (!impl_->completion_waiter && impl_->device.device != VK_NULL_HANDLE) {
    impl_->completion_waiter = std::make_unique<detail::completion_waiter>(
      impl_->device.device, impl_->compute_queue, impl_->queue_synchronization.state(impl_->compute_queue));
  }
  if (!impl_->completion_waiter) {
    error failure = make_error(errc::invalid_argument, "completion waiter requires a VkDevice");
    status result = fail(failure);
    // From here on, ownership of the sync objects is consumed and no error/status
    // construction is allowed to occur before completion is delivered.
    if (fence != VK_NULL_HANDLE) {
      (void)detail::wait_for_retirement(device(), 1, &fence);
    } else if (compute_queue() != VK_NULL_HANDLE) {
      auto const lock = lock_queue(compute_queue());
      (void)detail::wait_for_queue_retirement(compute_queue());
    } else if (semaphore != VK_NULL_HANDLE) {
      std::terminate();
    }
    if (semaphore != VK_NULL_HANDLE) { vkDestroySemaphore(device(), semaphore, nullptr); }
    if (fence != VK_NULL_HANDLE) { vkDestroyFence(device(), fence, nullptr); }
    if (on_done) { on_done(std::move(failure), false); }
    return result;
  }
  return impl_->completion_waiter->enqueue(semaphore, fence, std::move(stop_requested), std::move(on_done));
}

auto context::do_enqueue_borrowed_fence_wait(VkFence fence, detail::stop_fn stop_requested, detail::done_fn on_done)
  -> status
{
  if (!impl_->completion_waiter && impl_->device.device != VK_NULL_HANDLE) {
    impl_->completion_waiter = std::make_unique<detail::completion_waiter>(
      impl_->device.device, impl_->compute_queue, impl_->queue_synchronization.state(impl_->compute_queue));
  }
  if (!impl_->completion_waiter) {
    error failure = make_error(errc::invalid_argument, "completion waiter requires a VkDevice");
    status result = fail(failure);
    if (on_done) { on_done(std::move(failure), false); }
    return result;
  }
  return impl_->completion_waiter->enqueue_borrowed(fence, std::move(stop_requested), std::move(on_done));
}

auto context::do_enqueue_host(detail::host_task_fn task) -> status
{
  if (!impl_->host_agent) { impl_->host_agent = std::make_unique<detail::host_agent>(); }
  return impl_->host_agent->enqueue(std::move(task));
}

auto detail::enqueue_host(context_handle const &state, host_task_fn task) -> status
{
  if (!state) { return fail(errc::invalid_argument, "scheduler has no context"); }
  auto facade = context_access::facade(state);
  return facade.enqueue_host(std::move(task));
}

auto detail::retirement_future(context_handle const &state) -> std::shared_future<void>
{ return state != nullptr ? state->retirement : std::shared_future<void>{}; }

auto context::allocate_command_buffer() -> result<VkCommandBuffer>
{ return allocate_command_buffer(compute_queue_ref()); }

auto context::allocate_command_buffer(queue_ref queue) -> result<VkCommandBuffer>
{
  if (queue.queue == VK_NULL_HANDLE || queue.family == VK_QUEUE_FAMILY_IGNORED) {
    return fail(errc::invalid_argument, "allocate_command_buffer requires a valid queue");
  }
  VKEXEC_TRY_ASSIGN(pool, acquire_command_pool(queue.family));
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = pool;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  VkCommandBuffer cmd{ VK_NULL_HANDLE };
  if (VkResult const result = vkAllocateCommandBuffers(impl_->device.device, &alloc_info, &cmd); result != VK_SUCCESS) {
    release_command_pool(queue.family, pool);
    return fail(result, "vkAllocateCommandBuffers failed");
  }
#if VKEXEC_HAS_EXCEPTIONS
  try {
#endif
    std::scoped_lock const lock(impl_->command_buffer_mutex);
    impl_->command_buffer_pools.emplace(cmd, std::pair{ queue.family, pool });
#if VKEXEC_HAS_EXCEPTIONS
  } catch (...) {
    vkFreeCommandBuffers(device(), pool, 1, &cmd);
    release_command_pool(queue.family, pool);
    return fail(unexpected_exception_error());
  }
#endif
  return cmd;
}

auto context::free_command_buffer(VkCommandBuffer cmd) -> void
{
  if (cmd == VK_NULL_HANDLE) { return; }
  VkCommandPool pool{ VK_NULL_HANDLE };
  std::uint32_t family{ VK_QUEUE_FAMILY_IGNORED };
  {
    std::scoped_lock const lock(impl_->command_buffer_mutex);
    auto found = impl_->command_buffer_pools.find(cmd);
    if (found == impl_->command_buffer_pools.end()) { return; }
    family = found->second.first;
    pool = found->second.second;
    impl_->command_buffer_pools.erase(found);
  }
  vkFreeCommandBuffers(device(), pool, 1, &cmd);
  release_command_pool(family, pool);
}

auto context::submit_and_wait(VkCommandBuffer cmd) -> status { return submit_and_wait(cmd, compute_queue_ref()); }

auto context::submit_and_wait(VkCommandBuffer cmd, queue_ref queue) -> status
{
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence{ VK_NULL_HANDLE };
  if (VkResult const create_result = vkCreateFence(impl_->device.device, &fence_info, nullptr, &fence);
    create_result != VK_SUCCESS) {
    return fail(create_result, "vkCreateFence failed");
  }

  std::array<VkCommandBuffer, 1> const commands{ cmd };
  auto submitted = submit(queue_submit{ .command_buffers = commands, .fence = fence, .queue = queue.queue });
  if (!submitted) {
    vkDestroyFence(impl_->device.device, fence, nullptr);
    return submitted;
  }
  if (VkResult const wait_result =
        detail::dispatch_wait_for_fences(impl_->device.device, 1, &fence, VK_TRUE, UINT64_MAX);
    wait_result != VK_SUCCESS) {
    // A failed wait does not prove that a submitted command buffer has retired.
    // The test dispatch can fail this call while real GPU work remains pending.
    // Keep the fence and command buffer alive unless the real wait confirms retirement.
    (void)detail::wait_for_retirement(impl_->device.device, 1, &fence);
    vkDestroyFence(impl_->device.device, fence, nullptr);
    return fail(wait_result, "vkWaitForFences failed");
  }
  vkDestroyFence(impl_->device.device, fence, nullptr);
  return {};
}

auto context::submit_async(VkCommandBuffer cmd, VkSemaphore *out_semaphore, VkFence *out_fence) -> status
{ return submit_async(cmd, out_semaphore, out_fence, compute_queue_ref()); }

auto context::submit_async(VkCommandBuffer cmd, VkSemaphore *out_semaphore, VkFence *out_fence, queue_ref queue)
  -> status
{
  if (out_semaphore == nullptr) { return fail(errc::invalid_argument, "submit_async requires out_semaphore"); }

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkSemaphore sem{ VK_NULL_HANDLE };
  if (VkResult const sem_result = vkCreateSemaphore(impl_->device.device, &semaphore_info, nullptr, &sem);
    sem_result != VK_SUCCESS) {
    return fail(sem_result, "vkCreateSemaphore failed");
  }

  VkFence fence{ VK_NULL_HANDLE };
  if (out_fence != nullptr) {
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (VkResult const fence_result = vkCreateFence(impl_->device.device, &fence_info, nullptr, &fence);
      fence_result != VK_SUCCESS) {
      vkDestroySemaphore(impl_->device.device, sem, nullptr);
      return fail(fence_result, "vkCreateFence failed");
    }
    *out_fence = fence;
  }

  std::array<VkCommandBuffer, 1> const commands{ cmd };
  std::array<semaphore_submit, 1> const signals{ semaphore_submit{ .semaphore = sem } };
  auto submitted =
    submit(queue_submit{ .command_buffers = commands, .signals = signals, .fence = fence, .queue = queue.queue });
  if (!submitted) {
    vkDestroySemaphore(impl_->device.device, sem, nullptr);
    if (fence != VK_NULL_HANDLE) { vkDestroyFence(impl_->device.device, fence, nullptr); }
    if (out_fence != nullptr) { *out_fence = VK_NULL_HANDLE; }
    return submitted;
  }
  *out_semaphore = sem;
  return {};
}

auto context::submit(queue_submit const &info) const -> status
{
  auto *queue = info.queue != VK_NULL_HANDLE ? info.queue : impl_->compute_queue;
  return detail::submit(info,
    queue,
    impl_->synchronization,
    impl_->capabilities.timeline_semaphore,
    impl_->procs,
    *impl_->queue_synchronization.state(queue));
}

}// namespace vkexec
