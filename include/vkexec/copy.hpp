#ifndef VKEXEC_COPY_HPP
#define VKEXEC_COPY_HPP

#include <vkexec/context.hpp>
#include <vkexec/detail/result.hpp>
#include <vkexec/gpu_buffer.hpp>

#include <vulkan/vulkan.h>

#include <cstddef>
#include <span>

namespace vkexec {

/// Record `vkCmdCopyBuffer` between two buffer handles.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
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
// NOLINTEND(bugprone-easily-swappable-parameters)

/// Host-write `bytes` into `staging`, copy to `device`, and block until complete.
[[nodiscard]] auto upload_to_device(context &ctx,
  gpu_buffer &staging,
  gpu_buffer const &device,
  std::span<std::byte const> bytes) -> detail::status;

}// namespace vkexec

#endif// VKEXEC_COPY_HPP
