#include <vkexec/context.hpp>
#include <vkexec/detail/config.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/pipeline_cache.hpp>
#include <vkexec_edsl/trace.hpp>
#include <vkexec_edsl/trace_access.hpp>

#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
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

  auto configure_instance_builder(vkb::InstanceBuilder &builder, bool validation_layers) -> void
  {
    builder.set_app_name("vkexec").set_engine_name("vkexec").require_api_version(1, 2);
    if (validation_layers) { builder.enable_validation_layers().use_default_debug_messenger(); }
  }

  auto build_headless_instance(bool validation_layers) -> vkb::Instance
  {
    vkb::InstanceBuilder builder{};
    configure_instance_builder(builder, validation_layers);
    builder.set_headless();
    return unwrap(builder.build(), "vk-bootstrap InstanceBuilder");
  }

  auto build_instance_with_extensions(std::vector<char const *> const &instance_extensions, bool validation_layers)
    -> vkb::Instance
  {
    vkb::InstanceBuilder builder{};
    configure_instance_builder(builder, validation_layers);
    builder.set_headless().enable_extensions(instance_extensions.size(), instance_extensions.data());
    return unwrap(builder.build(), "vk-bootstrap InstanceBuilder");
  }

}// namespace

context::context(scheduler_options opts)
  : instance_(build_headless_instance(opts.validation_layers)),
    physical_device_(
      unwrap(vkb::PhysicalDeviceSelector{ instance_ }.set_minimum_version(1, 2).require_present(false).select(),
        "vk-bootstrap PhysicalDeviceSelector")),
    device_(unwrap(vkb::DeviceBuilder{ physical_device_ }.build(), "vk-bootstrap DeviceBuilder")), has_instance_(true),
    has_device_(true), owns_instance_(true), owns_device_(true), owns_allocator_(true)
{
  fetch_queues(false);
  create_command_pool();
  create_allocator();
  pipeline_cache_ = std::make_unique<pipeline_cache>(*this);
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
}

context::context(instance_only_tag tag, scheduler_options opts, std::vector<char const *> const &instance_extensions)
  : instance_(build_instance_with_extensions(instance_extensions, opts.validation_layers)), has_instance_(true),
    owns_instance_(true)
{ (void)tag; }

auto context::complete_for_surface(VkSurfaceKHR surface) -> void
{
  if (surface == VK_NULL_HANDLE) { VKEXEC_THROW(std::invalid_argument("complete_for_surface requires a surface")); }

  physical_device_ =
    unwrap(vkb::PhysicalDeviceSelector{ instance_ }.set_surface(surface).set_minimum_version(1, 2).select(),
      "vk-bootstrap PhysicalDeviceSelector");
  device_ = unwrap(vkb::DeviceBuilder{ physical_device_ }.build(), "vk-bootstrap DeviceBuilder");
  has_device_ = true;
  owns_device_ = true;
  owns_allocator_ = true;
  fetch_queues(true);
  create_command_pool();
  create_allocator();
  pipeline_cache_ = std::make_unique<pipeline_cache>(*this);
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
  allocator_info.vulkanApiVersion = VK_API_VERSION_1_2;
  if (vmaCreateAllocator(&allocator_info, &allocator_) != VK_SUCCESS) { fail("vmaCreateAllocator failed"); }
}

auto context::lock_host() const -> std::unique_lock<std::mutex> { return std::unique_lock{ host_mutex_ }; }

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
