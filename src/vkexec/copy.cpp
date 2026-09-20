#include <vkexec/copy.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/gpu_buffer.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace vkexec {

auto upload_to_device(context &ctx, gpu_buffer &staging, gpu_buffer const &device, std::span<std::byte const> bytes)
  -> status
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

  // One-shot transfer: host->staging visibility, copy, then submit_and_wait.
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

auto download_to_host(context &ctx, gpu_buffer &staging, gpu_buffer const &device, std::span<std::byte> out) -> status
{
  if (staging.memory() != gpu_buffer_memory::staging && staging.memory() != gpu_buffer_memory::host_visible) {
    return fail(errc::invalid_argument, "download_to_host staging buffer must be host-visible");
  }
  if (device.memory() != gpu_buffer_memory::device_local) {
    return fail(errc::invalid_argument, "download_to_host source must be device-local");
  }
  if (out.size() > staging.size() || out.size() > device.size()) {
    return fail(errc::out_of_range, "download_to_host byte span exceeds buffer size");
  }

  auto staging_map = staging.mapped();
  if (staging_map.size() < out.size()) { return fail(errc::out_of_range, "download_to_host staging map is too small"); }

  VKEXEC_TRY_ASSIGN(cmd, ctx.allocate_command_buffer());

  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
    ctx.free_command_buffer(cmd);
    return fail(errc::unsupported, "vkBeginCommandBuffer failed for download_to_host");
  }

  VkBufferMemoryBarrier device_barrier{};
  device_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  device_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  device_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  device_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  device_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  device_barrier.buffer = device.handle();
  device_barrier.offset = 0;
  device_barrier.size = static_cast<VkDeviceSize>(out.size());

  VkBufferMemoryBarrier staging_barrier{};
  staging_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  staging_barrier.srcAccessMask = 0;
  staging_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  staging_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  staging_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  staging_barrier.buffer = staging.handle();
  staging_barrier.offset = 0;
  staging_barrier.size = static_cast<VkDeviceSize>(out.size());

  std::array<VkBufferMemoryBarrier, 2> before{ device_barrier, staging_barrier };
  vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    0,
    0,
    nullptr,
    static_cast<std::uint32_t>(before.size()),
    before.data(),
    0,
    nullptr);

  cmd_copy_buffer(cmd, device.handle(), staging.handle(), static_cast<VkDeviceSize>(out.size()));

  VkBufferMemoryBarrier host_barrier{};
  host_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  host_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  host_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  host_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  host_barrier.buffer = staging.handle();
  host_barrier.offset = 0;
  host_barrier.size = static_cast<VkDeviceSize>(out.size());

  vkCmdPipelineBarrier(
    cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &host_barrier, 0, nullptr);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    ctx.free_command_buffer(cmd);
    return fail(errc::unsupported, "vkEndCommandBuffer failed for download_to_host");
  }

  auto submitted = ctx.submit_and_wait(cmd);
  ctx.free_command_buffer(cmd);
  if (!submitted) { return submitted; }

  std::memcpy(out.data(), staging_map.data(), out.size());
  return {};
}

}// namespace vkexec
