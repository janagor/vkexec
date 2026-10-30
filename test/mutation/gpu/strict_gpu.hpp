#ifndef VKEXEC_MUTATION_STRICT_GPU_HPP
#define VKEXEC_MUTATION_STRICT_GPU_HPP
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <vkexec/context.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_vma/resources.hpp>

namespace mutation_gpu {
[[noreturn]] inline void infrastructure(char const *message)
{
  std::fprintf(stderr, "GPU_INFRASTRUCTURE_FAILURE: %s\n", message);
  std::exit(86);
}
template<class Sender> auto wait(Sender &&sender)
{
  auto result = vkexec::try_sync_wait_value(std::forward<Sender>(sender));
  if (!result) { infrastructure(result.error().message().c_str()); }
  return std::move(*result);
}
inline auto identity(VkPhysicalDevice device) -> std::string
{
  VkPhysicalDeviceIDProperties ids{};
  ids.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
  VkPhysicalDeviceProperties2 properties{};
  properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties.pNext = &ids;
  vkGetPhysicalDeviceProperties2(device, &properties);
  auto const &p = properties.properties;
  std::ostringstream out;
  out << p.vendorID << ':' << p.deviceID << ':' << p.apiVersion << ':' << p.driverVersion << ':';
  for (auto byte : ids.deviceUUID) { out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte); }
  out << ':';
  for (auto byte : ids.driverUUID) { out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte); }
  return out.str();
}
inline auto context(bool inspect_software = false) -> std::unique_ptr<vkexec::context>
{
  auto ctx = wait(vkexec::factory::make_context());
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(ctx->physical_device(), &properties);
  if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU && !inspect_software
      && !std::getenv("VKEXEC_GPU_ALLOW_SOFTWARE")) {
    infrastructure("software device rejected; hardware measurement inconclusive");
  }
  auto const *expected = std::getenv("VKEXEC_GPU_IDENTITY");
  if (expected && identity(ctx->physical_device()) != expected) { infrastructure("selected device changed"); }
  std::uint32_t count{};
  vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device(), &count, nullptr);
  std::vector<VkQueueFamilyProperties> queues(count);
  vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device(), &count, queues.data());
  if (ctx->queue_family() >= count || queues.at(ctx->queue_family()).queueCount == 0
      || !(queues.at(ctx->queue_family()).queueFlags & VK_QUEUE_COMPUTE_BIT)) {
    infrastructure("compute/transfer queue unavailable");
  }
  return ctx;
}
}// namespace mutation_gpu
#endif
