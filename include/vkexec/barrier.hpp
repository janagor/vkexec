#ifndef VKEXEC_BARRIER_HPP
#define VKEXEC_BARRIER_HPP

#include <vulkan/vulkan.h>

#include <cstdint>

namespace vkexec {

struct memory_barrier_params
{
  VkPipelineStageFlags src_stage{};
  VkPipelineStageFlags dst_stage{};
  VkAccessFlags src_access{};
  VkAccessFlags dst_access{};
};

inline auto memory_barrier(VkCommandBuffer cmd, memory_barrier_params params) -> void
{
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = params.src_access;
  barrier.dstAccessMask = params.dst_access;
  vkCmdPipelineBarrier(cmd, params.src_stage, params.dst_stage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

struct image_barrier_params
{
  VkImage image{ VK_NULL_HANDLE };
  VkImageAspectFlags aspect{ VK_IMAGE_ASPECT_COLOR_BIT };
  VkImageLayout old_layout{ VK_IMAGE_LAYOUT_UNDEFINED };
  VkImageLayout new_layout{ VK_IMAGE_LAYOUT_GENERAL };
  VkPipelineStageFlags src_stage{ VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT };
  VkPipelineStageFlags dst_stage{ VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT };
  VkAccessFlags src_access{ 0 };
  VkAccessFlags dst_access{ 0 };
};

inline auto image_barrier(VkCommandBuffer cmd, image_barrier_params params) -> void
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

  [[nodiscard]] inline auto flags(std::uint32_t bits) -> VkFlags { return static_cast<VkFlags>(bits); }

  struct transfer_to_compute_t
  {
    auto operator()(VkCommandBuffer cmd) const -> void
    {
      memory_barrier(cmd,
        {
          .src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT,
          .dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          .src_access = VK_ACCESS_TRANSFER_WRITE_BIT,
          .dst_access = flags(static_cast<std::uint32_t>(VK_ACCESS_SHADER_READ_BIT)
                              | static_cast<std::uint32_t>(VK_ACCESS_SHADER_WRITE_BIT)),
        });
    }
  };

  struct compute_to_compute_t
  {
    auto operator()(VkCommandBuffer cmd) const -> void
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
  };

  struct compute_to_graphics_t
  {
    auto operator()(VkCommandBuffer cmd) const -> void
    {
      memory_barrier(cmd,
        {
          .src_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          .dst_stage = flags(static_cast<std::uint32_t>(VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT)
                             | static_cast<std::uint32_t>(VK_PIPELINE_STAGE_VERTEX_SHADER_BIT)
                             | static_cast<std::uint32_t>(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)),
          .src_access = VK_ACCESS_SHADER_WRITE_BIT,
          .dst_access = flags(static_cast<std::uint32_t>(VK_ACCESS_INDIRECT_COMMAND_READ_BIT)
                              | static_cast<std::uint32_t>(VK_ACCESS_SHADER_READ_BIT)),
        });
    }
  };

  struct graphics_to_compute_t
  {
    auto operator()(VkCommandBuffer cmd) const -> void
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
  };

  struct compute_read_t
  {
    auto operator()(VkCommandBuffer cmd) const -> void
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
  };

  [[nodiscard]] inline auto transfer_to_compute() -> transfer_to_compute_t { return {}; }
  [[nodiscard]] inline auto compute_to_compute() -> compute_to_compute_t { return {}; }
  [[nodiscard]] inline auto compute_to_graphics() -> compute_to_graphics_t { return {}; }
  [[nodiscard]] inline auto graphics_to_compute() -> graphics_to_compute_t { return {}; }
  [[nodiscard]] inline auto compute_read() -> compute_read_t { return {}; }

  inline auto transfer_to_compute(VkCommandBuffer cmd) -> void { transfer_to_compute()(cmd); }
  inline auto compute_to_compute(VkCommandBuffer cmd) -> void { compute_to_compute()(cmd); }
  inline auto compute_to_graphics(VkCommandBuffer cmd) -> void { compute_to_graphics()(cmd); }
  inline auto graphics_to_compute(VkCommandBuffer cmd) -> void { graphics_to_compute()(cmd); }
  inline auto compute_read(VkCommandBuffer cmd) -> void { compute_read()(cmd); }

}// namespace barrier

}// namespace vkexec

#endif// VKEXEC_BARRIER_HPP
