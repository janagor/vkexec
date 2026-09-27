#ifndef VKEXEC_COPY_HPP
#define VKEXEC_COPY_HPP

//! \file
//! Buffer copy command-recording helpers.

#include <vulkan/vulkan.h>

namespace vkexec {

/**
 * Records `vkCmdCopyBuffer` for a single region between two buffer handles.
 *
 * @param cmd Command buffer currently in the recording state.
 * @param src Source buffer handle.
 * @param dst Destination buffer handle.
 * @param size Number of bytes to copy.
 * @param src_offset Byte offset in `src`.
 * @param dst_offset Byte offset in `dst`.
 */
inline auto cmd_copy_buffer(VkCommandBuffer cmd,
  VkBuffer src,
  VkBuffer dst,
  VkDeviceSize size,
  VkDeviceSize src_offset = 0,
  VkDeviceSize dst_offset = 0) -> void
{
  VkBufferCopy region{};
  region.srcOffset = src_offset;
  region.dstOffset = dst_offset;
  region.size = size;
  vkCmdCopyBuffer(cmd, src, dst, 1, &region);
}

}// namespace vkexec

#endif// VKEXEC_COPY_HPP
