#include <vkexec/context.hpp>
#include <vkexec/scheduler.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace vkexec {
namespace {

[[noreturn]] void fail(std::string const &what)
{
  throw std::runtime_error(what);
}

template<typename T>
T unwrap(vkb::Result<T> result, char const *what)
{
  if (!result) {
    fail(std::string(what) + ": " + result.error().message() + " (" + std::to_string(result.vk_result()) + ")");
  }
  return result.value();
}

} // namespace

context::context()
{
  auto builder = vkb::InstanceBuilder{}
                   .set_app_name("vkexec")
                   .set_engine_name("vkexec")
                   .require_api_version(1, 2)
                   .set_headless();
  instance_ = unwrap(builder.build(), "vk-bootstrap InstanceBuilder");
  has_instance_ = true;

  auto selector =
    vkb::PhysicalDeviceSelector{ instance_ }.set_minimum_version(1, 2).require_present(false);
  physical_device_ = unwrap(selector.select(), "vk-bootstrap PhysicalDeviceSelector");

  device_ = unwrap(vkb::DeviceBuilder{ physical_device_ }.build(), "vk-bootstrap DeviceBuilder");
  has_device_ = true;
  fetch_queues(false);
  create_command_pool();
  create_allocator();
  pipeline_cache_ = std::make_unique<PipelineCache>(*this);
}

context::context(instance_only_tag /*tag*/, std::vector<const char *> instance_extensions)
{
  auto builder = vkb::InstanceBuilder{}
                   .set_app_name("vkexec")
                   .set_engine_name("vkexec")
                   .require_api_version(1, 2)
                   .set_headless()
                   .enable_extensions(instance_extensions.size(), instance_extensions.data());
  instance_ = unwrap(builder.build(), "vk-bootstrap InstanceBuilder");
  has_instance_ = true;
}

void context::complete_for_surface(VkSurfaceKHR surface)
{
  if (surface == VK_NULL_HANDLE) { throw std::invalid_argument("complete_for_surface requires a surface"); }

  auto selector = vkb::PhysicalDeviceSelector{ instance_ }.set_surface(surface).set_minimum_version(1, 2);
  physical_device_ = unwrap(selector.select(), "vk-bootstrap PhysicalDeviceSelector");

  device_ = unwrap(vkb::DeviceBuilder{ physical_device_ }.build(), "vk-bootstrap DeviceBuilder");
  has_device_ = true;
  fetch_queues(true);
  create_command_pool();
  create_allocator();
  pipeline_cache_ = std::make_unique<PipelineCache>(*this);
  presentation_enabled_ = true;
}

void context::fetch_queues(bool want_present)
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
  if (has_device_) {
    vkDeviceWaitIdle(device_.device);
    if (allocator_ != VK_NULL_HANDLE) {
      vmaDestroyAllocator(allocator_);
      allocator_ = VK_NULL_HANDLE;
    }
    if (command_pool_ != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device_.device, command_pool_, nullptr);
      command_pool_ = VK_NULL_HANDLE;
    }
    vkb::destroy_device(device_);
    has_device_ = false;
  }
  if (has_instance_) {
    vkb::destroy_instance(instance_);
    has_instance_ = false;
  }
}

void context::create_command_pool()
{
  VkCommandPoolCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pci.queueFamilyIndex = queue_family_;
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  if (vkCreateCommandPool(device_.device, &pci, nullptr, &command_pool_) != VK_SUCCESS) {
    fail("vkCreateCommandPool failed");
  }
}

void context::create_allocator()
{
  VmaAllocatorCreateInfo aci{};
  aci.physicalDevice = physical_device_.physical_device;
  aci.device = device_.device;
  aci.instance = instance_.instance;
  aci.vulkanApiVersion = VK_API_VERSION_1_2;
  if (vmaCreateAllocator(&aci, &allocator_) != VK_SUCCESS) { fail("vmaCreateAllocator failed"); }
}

VkCommandBuffer context::allocate_command_buffer()
{
  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = command_pool_;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cmd{ VK_NULL_HANDLE };
  if (vkAllocateCommandBuffers(device_.device, &ai, &cmd) != VK_SUCCESS) { fail("vkAllocateCommandBuffers failed"); }
  return cmd;
}

void context::free_command_buffer(VkCommandBuffer cmd)
{
  vkFreeCommandBuffers(device_.device, command_pool_, 1, &cmd);
}

void context::submit_and_wait(VkCommandBuffer cmd)
{
  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence{ VK_NULL_HANDLE };
  if (vkCreateFence(device_.device, &fci, nullptr, &fence) != VK_SUCCESS) { fail("vkCreateFence failed"); }

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  if (vkQueueSubmit(compute_queue_, 1, &si, fence) != VK_SUCCESS) {
    vkDestroyFence(device_.device, fence, nullptr);
    fail("vkQueueSubmit failed");
  }
  if (vkWaitForFences(device_.device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
    vkDestroyFence(device_.device, fence, nullptr);
    fail("vkWaitForFences failed");
  }
  vkDestroyFence(device_.device, fence, nullptr);
}

VkSemaphore context::submit_async(VkCommandBuffer cmd, VkFence *out_fence)
{
  VkSemaphoreCreateInfo sci{};
  sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkSemaphore sem{ VK_NULL_HANDLE };
  if (vkCreateSemaphore(device_.device, &sci, nullptr, &sem) != VK_SUCCESS) { fail("vkCreateSemaphore failed"); }

  VkFence fence{ VK_NULL_HANDLE };
  if (out_fence != nullptr) {
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device_.device, &fci, nullptr, &fence) != VK_SUCCESS) {
      vkDestroySemaphore(device_.device, sem, nullptr);
      fail("vkCreateFence failed");
    }
    *out_fence = fence;
  }

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  si.signalSemaphoreCount = 1;
  si.pSignalSemaphores = &sem;
  if (vkQueueSubmit(compute_queue_, 1, &si, fence) != VK_SUCCESS) {
    vkDestroySemaphore(device_.device, sem, nullptr);
    fail("vkQueueSubmit failed");
  }
  return sem;
}

} // namespace vkexec
