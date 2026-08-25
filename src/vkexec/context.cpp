#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_edsl/trace.hpp>

#include "detail/completion_waiter.hpp"
#include "detail/host_agent.hpp"
#include "pipeline_cache.hpp"
#include "vkexec_edsl/trace_access.hpp"

#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace vkexec {
namespace {

  [[noreturn]] auto fail(std::string const &what) -> void { VKEXEC_THROW(std::runtime_error(what)); }

  template<typename T> auto unwrap(vkb::Result<T> result, char const *what) -> T
  {
    if (!result) {
      fail(std::string(what) + ": " + result.error().message() + " (" + std::to_string(result.vk_result()) + ")");
    }
    return result.value();
  }

  auto resolve_api_version(vulkan_requirements const &requirements) -> std::uint32_t
  {
    std::uint32_t major = requirements.api_version_major;
    std::uint32_t minor = requirements.api_version_minor;
    if (major < vulkan_library::k_min_api_version_major
        || (major == vulkan_library::k_min_api_version_major && minor < vulkan_library::k_min_api_version_minor)) {
      major = vulkan_library::k_min_api_version_major;
      minor = vulkan_library::k_min_api_version_minor;
    }
    return VK_MAKE_API_VERSION(0, major, minor, 0);
  }

  auto append_unique(std::vector<char const *> &dst, std::span<char const * const> src) -> void
  {
    for (char const *extension : src) {
      if (extension == nullptr) { continue; }
      bool const exists = std::ranges::any_of(dst, [extension](char const *item) -> bool {
        return item != nullptr && std::strcmp(item, extension) == 0;
      });
      if (!exists) { dst.push_back(extension); }
    }
  }

  auto merge_instance_extensions(vulkan_requirements const &requirements,
    std::span<char const * const> extra_instance_extensions) -> std::vector<char const *>
  {
    std::vector<char const *> merged;
    append_unique(merged, vulkan_library::required_instance_extensions());
    append_unique(merged, requirements.instance_extensions);
    append_unique(merged, extra_instance_extensions);
    return merged;
  }

  auto merge_device_extensions(vulkan_requirements const &requirements, bool want_present) -> std::vector<char const *>
  {
    std::vector<char const *> merged;
    append_unique(merged, vulkan_library::required_device_extensions());
    if (want_present) { append_unique(merged, vulkan_library::required_presentation_device_extensions()); }
    append_unique(merged, requirements.device_extensions);
    return merged;
  }

  auto merge_features(vulkan_requirements const &requirements) -> VkPhysicalDeviceFeatures
  {
    // Library baseline is currently empty; user bits are required as requested.
    (void)vulkan_library::required_features();
    return requirements.features;
  }

  auto configure_instance_builder(vkb::InstanceBuilder &builder,
    scheduler_options const &opts,
    std::uint32_t api_version,
    std::span<char const * const> extra_instance_extensions) -> void
  {
    builder.set_app_name("vkexec")
      .set_engine_name("vkexec")
      .require_api_version(VK_API_VERSION_MAJOR(api_version), VK_API_VERSION_MINOR(api_version));
    if (opts.validation_layers) { builder.enable_validation_layers().use_default_debug_messenger(); }

    auto const instance_exts = merge_instance_extensions(opts.requirements, extra_instance_extensions);
    if (!instance_exts.empty()) { builder.enable_extensions(instance_exts.size(), instance_exts.data()); }
  }

  auto configure_device_selector(vkb::PhysicalDeviceSelector &selector,
    vulkan_requirements const &requirements,
    std::uint32_t api_version,
    bool want_present) -> void
  {
    selector.set_minimum_version(VK_API_VERSION_MAJOR(api_version), VK_API_VERSION_MINOR(api_version))
      .require_present(want_present);

    auto const device_exts = merge_device_extensions(requirements, want_present);
    if (!device_exts.empty()) { selector.add_required_extensions(device_exts.size(), device_exts.data()); }

    selector.set_required_features(merge_features(requirements));
  }

  auto build_headless_instance(scheduler_options const &opts, std::uint32_t api_version) -> vkb::Instance
  {
    vkb::InstanceBuilder builder{};
    configure_instance_builder(builder, opts, api_version, {});
    builder.set_headless();
    return unwrap(builder.build(), "vk-bootstrap InstanceBuilder");
  }

  auto build_instance_with_extensions(scheduler_options const &opts,
    std::uint32_t api_version,
    std::vector<char const *> const &extra_instance_extensions) -> vkb::Instance
  {
    vkb::InstanceBuilder builder{};
    configure_instance_builder(builder, opts, api_version, extra_instance_extensions);
    builder.set_headless();
    return unwrap(builder.build(), "vk-bootstrap InstanceBuilder");
  }

  auto select_physical_device(vkb::Instance const &instance,
    vulkan_requirements const &requirements,
    std::uint32_t api_version,
    VkSurfaceKHR surface,
    bool want_present) -> vkb::PhysicalDevice
  {
    vkb::PhysicalDeviceSelector selector{ instance };
    configure_device_selector(selector, requirements, api_version, want_present);
    if (surface != VK_NULL_HANDLE) { selector.set_surface(surface); }
    return unwrap(selector.select(), "vk-bootstrap PhysicalDeviceSelector");
  }

}// namespace

context::context(scheduler_options opts)
  : requirements_(opts.requirements), api_version_(resolve_api_version(requirements_)),
    instance_(build_headless_instance(opts, api_version_)),
    physical_device_(select_physical_device(instance_, requirements_, api_version_, VK_NULL_HANDLE, false)),
    device_(unwrap(vkb::DeviceBuilder{ physical_device_ }.build(), "vk-bootstrap DeviceBuilder")), has_instance_(true),
    has_device_(true), owns_instance_(true), owns_device_(true), owns_allocator_(true)
{
  fetch_queues(false);
  create_command_pool();
  create_allocator();
  pipeline_cache_ = std::make_unique<pipeline_cache>(*this);
  completion_waiter_ = std::make_unique<detail::completion_waiter>(device_.device, compute_queue_);
  host_agent_ = std::make_unique<detail::host_agent>();
}

auto context::adopt(context_adopt_info const &info) -> std::unique_ptr<context>
{ return std::unique_ptr<context>(new context(info)); }

context::context(context_adopt_info const &info)
{
  if (info.device == VK_NULL_HANDLE) { fail("context::adopt requires a VkDevice"); }
  if (info.compute_queue == VK_NULL_HANDLE) { fail("context::adopt requires a compute VkQueue"); }

  instance_.instance = info.instance;
  physical_device_.physical_device = info.physical_device;
  device_.device = info.device;
  has_instance_ = info.instance != VK_NULL_HANDLE;
  has_device_ = true;

  compute_queue_ = info.compute_queue;
  queue_family_ = info.compute_queue_family;

  if (info.graphics_queue != VK_NULL_HANDLE) {
    graphics_queue_ = info.graphics_queue;
    graphics_family_ = info.graphics_queue_family;
  } else {
    graphics_queue_ = compute_queue_;
    graphics_family_ = queue_family_;
  }

  if (info.present_queue != VK_NULL_HANDLE) {
    present_queue_ = info.present_queue;
    present_family_ = info.present_queue_family;
  } else {
    present_queue_ = graphics_queue_;
    present_family_ = graphics_family_;
  }

  if (info.allocator != VK_NULL_HANDLE) {
    allocator_ = info.allocator;
  } else {
    if (info.instance == VK_NULL_HANDLE || info.physical_device == VK_NULL_HANDLE) {
      fail("context::adopt requires instance and physical_device when allocator is null");
    }
    create_allocator();
    owns_allocator_ = true;
  }

  create_command_pool();
  pipeline_cache_ = std::make_unique<pipeline_cache>(*this);
  completion_waiter_ = std::make_unique<detail::completion_waiter>(device_.device, compute_queue_);
  host_agent_ = std::make_unique<detail::host_agent>();
}

context::context(instance_only_tag tag, scheduler_options opts, std::vector<char const *> const &instance_extensions)
  : requirements_(opts.requirements), api_version_(resolve_api_version(requirements_)),
    instance_(build_instance_with_extensions(opts, api_version_, instance_extensions)), has_instance_(true),
    owns_instance_(true)
{ (void)tag; }

auto context::complete_for_surface(VkSurfaceKHR surface) -> void
{
  if (surface == VK_NULL_HANDLE) { VKEXEC_THROW(std::invalid_argument("complete_for_surface requires a surface")); }

  physical_device_ = select_physical_device(instance_, requirements_, api_version_, surface, true);
  device_ = unwrap(vkb::DeviceBuilder{ physical_device_ }.build(), "vk-bootstrap DeviceBuilder");
  has_device_ = true;
  owns_device_ = true;
  owns_allocator_ = true;
  fetch_queues(true);
  create_command_pool();
  create_allocator();
  pipeline_cache_ = std::make_unique<pipeline_cache>(*this);
  completion_waiter_ = std::make_unique<detail::completion_waiter>(device_.device, compute_queue_);
  host_agent_ = std::make_unique<detail::host_agent>();
  presentation_enabled_ = true;
}

auto context::fetch_queues(bool want_present) -> void
{
  if (auto graphics = device_.get_queue_and_index(vkb::QueueType::graphics)) {
    graphics_queue_ = graphics->first;
    graphics_family_ = graphics->second;
  }

  if (auto compute = device_.get_queue_and_index(vkb::QueueType::compute)) {
    compute_queue_ = compute->first;
    queue_family_ = compute->second;
  } else if (graphics_queue_ != VK_NULL_HANDLE) {
    // Graphics queues are compute-capable on typical GPUs.
    compute_queue_ = graphics_queue_;
    queue_family_ = graphics_family_;
  } else {
    fail("no compute or graphics queue available");
  }

  if (graphics_queue_ == VK_NULL_HANDLE) {
    graphics_queue_ = compute_queue_;
    graphics_family_ = queue_family_;
  }

  if (want_present) {
    auto present = device_.get_queue_and_index(vkb::QueueType::present);
    if (!present) { fail("no present queue available"); }
    present_queue_ = present->first;
    present_family_ = present->second;
  } else {
    present_queue_ = graphics_queue_;
    present_family_ = graphics_family_;
  }
}

context::~context()
{
  // Drain host schedule completions and fence waits before tearing down the device.
  host_agent_.reset();
  completion_waiter_.reset();
  pipeline_cache_.reset();

  if (device_.device != VK_NULL_HANDLE) {
    if (owns_device_) { vkDeviceWaitIdle(device_.device); }
    if (command_pool_ != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device_.device, command_pool_, nullptr);
      command_pool_ = VK_NULL_HANDLE;
    }
    if (owns_allocator_ && allocator_ != VK_NULL_HANDLE) {
      vmaDestroyAllocator(allocator_);
      allocator_ = VK_NULL_HANDLE;
    }
    if (owns_device_) {
      vkb::destroy_device(device_);
      has_device_ = false;
    }
  }

  if (owns_instance_ && has_instance_) {
    vkb::destroy_instance(instance_);
    has_instance_ = false;
  }
}

auto context::create_command_pool() -> void
{
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = queue_family_;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  if (vkCreateCommandPool(device_.device, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
    fail("vkCreateCommandPool failed");
  }
}

auto context::create_allocator() -> void
{
  VmaAllocatorCreateInfo allocator_info{};
  allocator_info.physicalDevice = physical_device_.physical_device;
  allocator_info.device = device_.device;
  allocator_info.instance = instance_.instance;
  allocator_info.vulkanApiVersion = api_version_;
  if (vmaCreateAllocator(&allocator_info, &allocator_) != VK_SUCCESS) { fail("vmaCreateAllocator failed"); }
}

auto context::lock_host() const -> std::unique_lock<std::mutex> { return std::unique_lock{ host_mutex_ }; }

auto context::ensure_completion_waiter() -> detail::completion_waiter &
{
  if (!completion_waiter_) {
    if (device_.device == VK_NULL_HANDLE) { fail("completion waiter requires a VkDevice"); }
    completion_waiter_ = std::make_unique<detail::completion_waiter>(device_.device, compute_queue_);
  }
  return *completion_waiter_;
}

auto context::ensure_host_agent() -> detail::host_agent &
{
  if (!host_agent_) { host_agent_ = std::make_unique<detail::host_agent>(); }
  return *host_agent_;
}

auto context::host_agent_thread_id() -> std::thread::id { return ensure_host_agent().thread_id(); }

auto context::do_enqueue_fence_wait(VkSemaphore semaphore,
  VkFence fence,
  std::move_only_function<bool()> stop_requested,
  std::move_only_function<void(std::exception_ptr, bool)> on_done) -> void
{ ensure_completion_waiter().enqueue(semaphore, fence, std::move(stop_requested), std::move(on_done)); }

auto context::do_enqueue_borrowed_fence_wait(VkFence fence,
  std::move_only_function<bool()> stop_requested,
  std::move_only_function<void(std::exception_ptr, bool)> on_done) -> void
{ ensure_completion_waiter().enqueue_borrowed(fence, std::move(stop_requested), std::move(on_done)); }

auto context::do_enqueue_host(std::move_only_function<void()> task) -> void
{ ensure_host_agent().enqueue(std::move(task)); }

auto context::allocate_command_buffer() -> VkCommandBuffer
{
  std::scoped_lock const lock(host_mutex_);
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = command_pool_;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  VkCommandBuffer cmd{ VK_NULL_HANDLE };
  if (vkAllocateCommandBuffers(device_.device, &alloc_info, &cmd) != VK_SUCCESS) {
    fail("vkAllocateCommandBuffers failed");
  }
  return cmd;
}

auto context::free_command_buffer(VkCommandBuffer cmd) -> void
{
  std::scoped_lock const lock(host_mutex_);
  vkFreeCommandBuffers(device_.device, command_pool_, 1, &cmd);
}

auto context::submit_and_wait(VkCommandBuffer cmd) -> void
{
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence{ VK_NULL_HANDLE };
  if (vkCreateFence(device_.device, &fence_info, nullptr, &fence) != VK_SUCCESS) { fail("vkCreateFence failed"); }

  {
    std::scoped_lock const lock(host_mutex_);
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    if (vkQueueSubmit(compute_queue_, 1, &submit_info, fence) != VK_SUCCESS) {
      vkDestroyFence(device_.device, fence, nullptr);
      fail("vkQueueSubmit failed");
    }
  }
  if (vkWaitForFences(device_.device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
    vkDestroyFence(device_.device, fence, nullptr);
    fail("vkWaitForFences failed");
  }
  vkDestroyFence(device_.device, fence, nullptr);
}

auto context::submit_async(VkCommandBuffer cmd, VkFence *out_fence) -> VkSemaphore
{
  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkSemaphore sem{ VK_NULL_HANDLE };
  if (vkCreateSemaphore(device_.device, &semaphore_info, nullptr, &sem) != VK_SUCCESS) {
    fail("vkCreateSemaphore failed");
  }

  VkFence fence{ VK_NULL_HANDLE };
  if (out_fence != nullptr) {
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device_.device, &fence_info, nullptr, &fence) != VK_SUCCESS) {
      vkDestroySemaphore(device_.device, sem, nullptr);
      fail("vkCreateFence failed");
    }
    *out_fence = fence;
  }

  {
    std::scoped_lock const lock(host_mutex_);
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &sem;
    if (vkQueueSubmit(compute_queue_, 1, &submit_info, fence) != VK_SUCCESS) {
      vkDestroySemaphore(device_.device, sem, nullptr);
      if (fence != VK_NULL_HANDLE) { vkDestroyFence(device_.device, fence, nullptr); }
      fail("vkQueueSubmit failed");
    }
  }
  return sem;
}

auto context::get_or_compile(edsl::trace_scope const &trace, std::uint32_t work_count) -> pipeline_resources &
{ return pipeline_cache_->get_or_compile(edsl::detail::trace_ast_access::get(trace), work_count); }

auto context::get_or_create_from_spirv(std::span<std::uint32_t const> spirv, layout_desc const &desc)
  -> pipeline_resources &
{ return pipeline_cache_->get_or_create_from_spirv(spirv, desc); }

}// namespace vkexec
