#include <vkexec/context.hpp>
#include <vkexec/scheduler.hpp>

#include <stdexcept>
#include <vector>

namespace vkexec {
namespace {

void check(VkResult result, const char *what)
{
  if (result != VK_SUCCESS) { throw std::runtime_error(what); }
}

} // namespace

context::context()
{
  create_instance();
  pick_device();
  create_device();
  create_command_pool();
  pipeline_cache_ = std::make_unique<PipelineCache>(*this);
}

context::~context()
{
  pipeline_cache_.reset();
  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);
    if (command_pool_ != VK_NULL_HANDLE) { vkDestroyCommandPool(device_, command_pool_, nullptr); }
    vkDestroyDevice(device_, nullptr);
  }
  if (instance_ != VK_NULL_HANDLE) { vkDestroyInstance(instance_, nullptr); }
}

void context::create_instance()
{
  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "vkexec";
  app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  app.pEngineName = "vkexec";
  app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
  app.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &app;
  check(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance failed");
}

void context::pick_device()
{
  std::uint32_t count = 0;
  check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices failed");
  if (count == 0) { throw std::runtime_error("no Vulkan physical devices"); }
  std::vector<VkPhysicalDevice> devices(count);
  check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()), "vkEnumeratePhysicalDevices failed");

  for (VkPhysicalDevice dev : devices) {
    std::uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> props(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, props.data());
    for (std::uint32_t i = 0; i < qcount; ++i) {
      if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        physical_ = dev;
        queue_family_ = i;
        return;
      }
    }
  }
  throw std::runtime_error("no compute queue family found");
}

void context::create_device()
{
  float priority = 1.0f;
  VkDeviceQueueCreateInfo qci{};
  qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci.queueFamilyIndex = queue_family_;
  qci.queueCount = 1;
  qci.pQueuePriorities = &priority;

  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  check(vkCreateDevice(physical_, &dci, nullptr, &device_), "vkCreateDevice failed");
  vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
}

void context::create_command_pool()
{
  VkCommandPoolCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pci.queueFamilyIndex = queue_family_;
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  check(vkCreateCommandPool(device_, &pci, nullptr, &command_pool_), "vkCreateCommandPool failed");
}

VkCommandBuffer context::allocate_command_buffer()
{
  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = command_pool_;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cmd{ VK_NULL_HANDLE };
  check(vkAllocateCommandBuffers(device_, &ai, &cmd), "vkAllocateCommandBuffers failed");
  return cmd;
}

void context::free_command_buffer(VkCommandBuffer cmd)
{
  vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
}

void context::submit_and_wait(VkCommandBuffer cmd)
{
  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence{ VK_NULL_HANDLE };
  check(vkCreateFence(device_, &fci, nullptr, &fence), "vkCreateFence failed");

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  check(vkQueueSubmit(queue_, 1, &si, fence), "vkQueueSubmit failed");
  check(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences failed");
  vkDestroyFence(device_, fence, nullptr);
}

VkSemaphore context::submit_async(VkCommandBuffer cmd, VkFence *out_fence)
{
  VkSemaphoreCreateInfo sci{};
  sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkSemaphore sem{ VK_NULL_HANDLE };
  check(vkCreateSemaphore(device_, &sci, nullptr, &sem), "vkCreateSemaphore failed");

  VkFence fence{ VK_NULL_HANDLE };
  if (out_fence != nullptr) {
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    check(vkCreateFence(device_, &fci, nullptr, &fence), "vkCreateFence failed");
    *out_fence = fence;
  }

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  si.signalSemaphoreCount = 1;
  si.pSignalSemaphores = &sem;
  check(vkQueueSubmit(queue_, 1, &si, fence), "vkQueueSubmit failed");
  return sem;
}

} // namespace vkexec
