#include <vkexec/barrier.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>

namespace vkexec {

auto memory_barrier(VkCommandBuffer cmd, memory_barrier_params params) -> void
{
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = params.src_access;
  barrier.dstAccessMask = params.dst_access;
  vkCmdPipelineBarrier(cmd, params.src_stage, params.dst_stage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

auto image_barrier(VkCommandBuffer cmd, image_barrier_params params) -> void
{
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcAccessMask = params.src_access;
  barrier.dstAccessMask = params.dst_access;
  barrier.oldLayout = params.old_layout;
  barrier.newLayout = params.new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = params.image;
  barrier.subresourceRange.aspectMask = params.aspect;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  vkCmdPipelineBarrier(cmd, params.src_stage, params.dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

namespace barrier {

auto transfer_to_compute_t::operator()(VkCommandBuffer cmd) const -> void
{
  memory_barrier(cmd,
    {
      .src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT,
      .dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      .src_access = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dst_access = flags(
        static_cast<std::uint32_t>(VK_ACCESS_SHADER_READ_BIT) | static_cast<std::uint32_t>(VK_ACCESS_SHADER_WRITE_BIT)),
    });
}

auto compute_to_compute_t::operator()(VkCommandBuffer cmd) const -> void
{
  memory_barrier(cmd,
    {
      .src_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      .dst_stage = flags(static_cast<std::uint32_t>(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
                         | static_cast<std::uint32_t>(VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT)),
      .src_access = VK_ACCESS_SHADER_WRITE_BIT,
      .dst_access = flags(static_cast<std::uint32_t>(VK_ACCESS_SHADER_READ_BIT)
                          | static_cast<std::uint32_t>(VK_ACCESS_SHADER_WRITE_BIT)
                          | static_cast<std::uint32_t>(VK_ACCESS_INDIRECT_COMMAND_READ_BIT)),
    });
}

auto compute_to_graphics_t::operator()(VkCommandBuffer cmd) const -> void
{
  memory_barrier(cmd,
    {
      .src_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      .dst_stage = flags(static_cast<std::uint32_t>(VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT)
                         | static_cast<std::uint32_t>(VK_PIPELINE_STAGE_VERTEX_SHADER_BIT)
                         | static_cast<std::uint32_t>(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)),
      .src_access = VK_ACCESS_SHADER_WRITE_BIT,
      .dst_access = flags(
        static_cast<std::uint32_t>(VK_ACCESS_INDIRECT_COMMAND_READ_BIT) | static_cast<std::uint32_t>(VK_ACCESS_SHADER_READ_BIT)),
    });
}

auto graphics_to_compute_t::operator()(VkCommandBuffer cmd) const -> void
{
  memory_barrier(cmd,
    {
      .src_stage = flags(static_cast<std::uint32_t>(VK_PIPELINE_STAGE_VERTEX_SHADER_BIT)
                         | static_cast<std::uint32_t>(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)),
      .dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      .src_access = VK_ACCESS_SHADER_READ_BIT,
      .dst_access = VK_ACCESS_SHADER_WRITE_BIT,
    });
}

auto compute_read_t::operator()(VkCommandBuffer cmd) const -> void
{
  memory_barrier(cmd,
    {
      .src_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      .dst_stage = flags(static_cast<std::uint32_t>(VK_PIPELINE_STAGE_VERTEX_SHADER_BIT)
                         | static_cast<std::uint32_t>(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)),
      .src_access = VK_ACCESS_SHADER_WRITE_BIT,
      .dst_access = VK_ACCESS_SHADER_READ_BIT,
    });
}

auto transfer_to_compute(VkCommandBuffer cmd) -> void { transfer_to_compute()(cmd); }
auto compute_to_compute(VkCommandBuffer cmd) -> void { compute_to_compute()(cmd); }
auto compute_to_graphics(VkCommandBuffer cmd) -> void { compute_to_graphics()(cmd); }
auto graphics_to_compute(VkCommandBuffer cmd) -> void { graphics_to_compute()(cmd); }
auto compute_read(VkCommandBuffer cmd) -> void { compute_read()(cmd); }

}// namespace barrier

}// namespace vkexec
