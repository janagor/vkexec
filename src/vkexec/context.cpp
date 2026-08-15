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
  create_instance({});
  pick_device(VK_NULL_HANDLE);
  create_device(false);
  create_command_pool();
  pipeline_cache_ = std::make_unique<PipelineCache>(*this);
}

context::context(instance_only_tag /*tag*/, std::vector<const char *> instance_extensions)
{
  create_instance(instance_extensions);
}

void context::complete_for_surface(VkSurfaceKHR surface)
{
  if (surface == VK_NULL_HANDLE) { throw std::invalid_argument("complete_for_surface requires a surface"); }
  pick_device(surface);
  create_device(true);
  create_command_pool();
  pipeline_cache_ = std::make_unique<PipelineCache>(*this);
  presentation_enabled_ = true;
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

void context::create_instance(const std::vector<const char *> &extra_extensions)
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
  ci.enabledExtensionCount = static_cast<std::uint32_t>(extra_extensions.size());
  ci.ppEnabledExtensionNames = extra_extensions.empty() ? nullptr : extra_extensions.data();
  check(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance failed");
}

void context::pick_device(VkSurfaceKHR surface)
{
  std::uint32_t count = 0;
  check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices failed");
  if (count == 0) { throw std::runtime_error("no Vulkan physical devices"); }
  std::vector<VkPhysicalDevice> devices(count);
  check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()), "vkEnumeratePhysicalDevices failed");

  // Prefer a single queue family that can do graphics+compute (+ present when needed) so one
  // command pool works for both bulk compute and window rendering.
  for (bool require_graphics : { true, false }) {
    if (surface != VK_NULL_HANDLE && !require_graphics) { continue; }
    for (VkPhysicalDevice dev : devices) {
      std::uint32_t qcount = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, nullptr);
      std::vector<VkQueueFamilyProperties> props(qcount);
      vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, props.data());

      for (std::uint32_t i = 0; i < qcount; ++i) {
        const bool has_graphics = (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0U;
        const bool has_compute = (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0U;
        if (!has_compute) { continue; }
        if (require_graphics && !has_graphics) { continue; }

        std::uint32_t present_family = i;
        if (surface != VK_NULL_HANDLE) {
          VkBool32 supported = VK_FALSE;
          vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &supported);
          if (supported != VK_TRUE) {
            int found_present = -1;
            for (std::uint32_t j = 0; j < qcount; ++j) {
              VkBool32 ok = VK_FALSE;
              vkGetPhysicalDeviceSurfaceSupportKHR(dev, j, surface, &ok);
              if (ok == VK_TRUE) {
                found_present = static_cast<int>(j);
                break;
              }
            }
            if (found_present < 0) { continue; }
            present_family = static_cast<std::uint32_t>(found_present);
          }
        }

        physical_ = dev;
        queue_family_ = i;
        graphics_family_ = i;
        present_family_ = present_family;
        return;
      }
    }
  }

  throw std::runtime_error(surface == VK_NULL_HANDLE ? "no compute queue family found"
                                                     : "no graphics/compute/present device found");
}

void context::create_device(bool enable_swapchain)
{
  std::vector<VkDeviceQueueCreateInfo> queue_infos;
  std::vector<float> priorities{ 1.0f };

  auto add_family = [&](std::uint32_t family) {
    for (const auto &existing : queue_infos) {
      if (existing.queueFamilyIndex == family) { return; }
    }
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = priorities.data();
    queue_infos.push_back(qci);
  };

  add_family(queue_family_);
  add_family(graphics_family_);
  add_family(present_family_);

  std::vector<const char *> device_extensions;
  if (enable_swapchain) { device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME); }

  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = static_cast<std::uint32_t>(queue_infos.size());
  dci.pQueueCreateInfos = queue_infos.data();
  dci.enabledExtensionCount = static_cast<std::uint32_t>(device_extensions.size());
  dci.ppEnabledExtensionNames = device_extensions.empty() ? nullptr : device_extensions.data();
  check(vkCreateDevice(physical_, &dci, nullptr, &device_), "vkCreateDevice failed");

  vkGetDeviceQueue(device_, queue_family_, 0, &compute_queue_);
  vkGetDeviceQueue(device_, graphics_family_, 0, &graphics_queue_);
  vkGetDeviceQueue(device_, present_family_, 0, &present_queue_);
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
  check(vkQueueSubmit(compute_queue_, 1, &si, fence), "vkQueueSubmit failed");
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
  check(vkQueueSubmit(compute_queue_, 1, &si, fence), "vkQueueSubmit failed");
  return sem;
}

} // namespace vkexec
