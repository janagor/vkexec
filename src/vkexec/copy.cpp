#include <vkexec/copy.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/result.hpp>
#include <vkexec/error.hpp>
#include <vkexec/gpu_buffer.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace vkexec {

auto upload_to_device(context &ctx, gpu_buffer &staging, gpu_buffer const &device, std::span<std::byte const> bytes)
  -> detail::status
{
  if (staging.memory() != gpu_buffer_memory::staging && staging.memory() != gpu_buffer_memory::host_visible) {
    return fail(errc::invalid_argument, "upload_to_device staging buffer must be host-visible");
  }
  if (device.memory() != gpu_buffer_memory::device_local) {
    return fail(errc::invalid_argument, "upload_to_device destination must be device-local");
  }
  if (bytes.size() > staging.size() || bytes.size() > device.size()) {
    return fail(errc::out_of_range, "upload_to_device byte span exceeds buffer size");
  }

  auto staging_map = staging.mapped();
  if (staging_map.size() < bytes.size()) {
    return fail(errc::out_of_range, "upload_to_device staging map is too small");
  }
  std::memcpy(staging_map.data(), bytes.data(), bytes.size());

  VKEXEC_TRY_ASSIGN(cmd, ctx.allocate_command_buffer());

  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
    ctx.free_command_buffer(cmd);
    return fail(errc::unsupported, "vkBeginCommandBuffer failed for upload_to_device");
  }

  VkBufferMemoryBarrier staging_barrier{};
  staging_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  staging_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
  staging_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  staging_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  staging_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  staging_barrier.buffer = staging.handle();
  staging_barrier.offset = 0;
  staging_barrier.size = static_cast<VkDeviceSize>(bytes.size());

  VkBufferMemoryBarrier device_barrier{};
  device_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  device_barrier.srcAccessMask = 0;
  device_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  device_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  device_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  device_barrier.buffer = device.handle();
  device_barrier.offset = 0;
  device_barrier.size = static_cast<VkDeviceSize>(bytes.size());

  std::array<VkBufferMemoryBarrier, 2> barriers{ staging_barrier, device_barrier };
  vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_HOST_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    0,
    0,
    nullptr,
    static_cast<std::uint32_t>(barriers.size()),
    barriers.data(),
    0,
    nullptr);

  cmd_copy_buffer(cmd, staging.handle(), device.handle(), static_cast<VkDeviceSize>(bytes.size()));

  barrier::transfer_to_compute(cmd);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    ctx.free_command_buffer(cmd);
    return fail(errc::unsupported, "vkEndCommandBuffer failed for upload_to_device");
  }

  auto const submitted = ctx.submit_and_wait(cmd);
  ctx.free_command_buffer(cmd);
  return submitted;
}

}// namespace vkexec
