#include "strict_gpu.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv)
{
  auto ctx = mutation_gpu::context(true);
  if (argc != 2) { mutation_gpu::infrastructure("expected device JSON output path"); }
  std::ofstream output(argv[1]);
  if (!output) { mutation_gpu::infrastructure("cannot write device JSON"); }
  std::uint32_t count{};
  if (vkEnumeratePhysicalDevices(ctx->instance(), &count, nullptr) != VK_SUCCESS || count == 0) {
    mutation_gpu::infrastructure("no Vulkan devices");
  }
  auto allocator = mutation_gpu::wait(vkexec::vma::factory::make_allocator(*ctx));
  auto allocation = mutation_gpu::wait(vkexec::allocate_buffer(allocator,
    vkexec::buffer_create_info{
      .size = 512, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .memory = vkexec::memory_domain::staging }));
  auto local = mutation_gpu::wait(vkexec::allocate_buffer(allocator,
    vkexec::buffer_create_info{ .size = 512,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .memory = vkexec::memory_domain::device_local }));
  if (allocation.mapped().size() != 512) { mutation_gpu::infrastructure("staging mapping unavailable"); }
  VkPhysicalDeviceProperties p{};
  vkGetPhysicalDeviceProperties(ctx->physical_device(), &p);
  char const *type = "other";
  switch (p.deviceType) {
  case VK_PHYSICAL_DEVICE_TYPE_CPU:
    type = "cpu";
    break;
  case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
    type = "integrated_gpu";
    break;
  case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
    type = "discrete_gpu";
    break;
  case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
    type = "virtual_gpu";
    break;
  default:
    break;
  }
  // std::quoted escapes quotes/backslashes; device names normally contain no controls.
  output << "{\"device_name\":" << std::quoted(p.deviceName) << ",\"vendor_id\":" << p.vendorID
         << ",\"device_id\":" << p.deviceID << ",\"device_type\":" << std::quoted(type) << ",\"api_version\":\""
         << VK_API_VERSION_MAJOR(p.apiVersion) << '.' << VK_API_VERSION_MINOR(p.apiVersion) << '.'
         << VK_API_VERSION_PATCH(p.apiVersion)
         << "\",\"driver_version\":" << std::quoted(std::to_string(p.driverVersion))
         << ",\"queue_family\":" << ctx->queue_family()
         << ",\"software_device\":" << (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? "true" : "false")
         << ",\"identity\":" << std::quoted(mutation_gpu::identity(ctx->physical_device())) << "}\n";
  output.close();
  if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU && !std::getenv("VKEXEC_GPU_ALLOW_SOFTWARE")) {
    mutation_gpu::infrastructure("software device rejected; hardware measurement inconclusive");
  }
}
