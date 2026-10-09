#ifndef VKEXEC_BARRIER_PARAMS_HPP
#define VKEXEC_BARRIER_PARAMS_HPP

#include <vulkan/vulkan.h>

#include <cstdint>

namespace vkexec {

//! Parameters for a global memory dependency recorded with `memory_barrier`.
struct memory_barrier_params
{
  VkPipelineStageFlags2 src_stage{ VK_PIPELINE_STAGE_2_NONE };
  VkPipelineStageFlags2 dst_stage{ VK_PIPELINE_STAGE_2_NONE };
  VkAccessFlags2 src_access{ VK_ACCESS_2_NONE };
  VkAccessFlags2 dst_access{ VK_ACCESS_2_NONE };
};

//! Parameters for a buffer dependency recorded with `buffer_barrier`.
struct buffer_barrier_params
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  VkDeviceSize offset{ 0 };
  VkDeviceSize size{ VK_WHOLE_SIZE };
  VkPipelineStageFlags2 src_stage{ VK_PIPELINE_STAGE_2_NONE };
  VkPipelineStageFlags2 dst_stage{ VK_PIPELINE_STAGE_2_NONE };
  VkAccessFlags2 src_access{ VK_ACCESS_2_NONE };
  VkAccessFlags2 dst_access{ VK_ACCESS_2_NONE };
  std::uint32_t src_queue_family{ VK_QUEUE_FAMILY_IGNORED };
  std::uint32_t dst_queue_family{ VK_QUEUE_FAMILY_IGNORED };
};

//! Parameters for an image dependency or layout transition.
struct image_barrier_params
{
  VkImage image{ VK_NULL_HANDLE };
  VkImageSubresourceRange range{
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = 1,
    .baseArrayLayer = 0,
    .layerCount = 1,
  };
  VkImageLayout old_layout{ VK_IMAGE_LAYOUT_UNDEFINED };
  VkImageLayout new_layout{ VK_IMAGE_LAYOUT_GENERAL };
  VkPipelineStageFlags2 src_stage{ VK_PIPELINE_STAGE_2_NONE };
  VkPipelineStageFlags2 dst_stage{ VK_PIPELINE_STAGE_2_NONE };
  VkAccessFlags2 src_access{ VK_ACCESS_2_NONE };
  VkAccessFlags2 dst_access{ VK_ACCESS_2_NONE };
  std::uint32_t src_queue_family{ VK_QUEUE_FAMILY_IGNORED };
  std::uint32_t dst_queue_family{ VK_QUEUE_FAMILY_IGNORED };
};

}// namespace vkexec

#endif// VKEXEC_BARRIER_PARAMS_HPP
