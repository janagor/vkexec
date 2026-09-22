#ifndef VKEXEC_COPY_HPP
#define VKEXEC_COPY_HPP

//! \file
//! Buffer copy helpers for command recording and host->device upload.

#include <vkexec/context.hpp>
#include <vkexec/gpu_buffer.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan.h>

#include <cstddef>
#include <span>

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

/**
 * Writes `bytes` into host-visible `staging`, copies to `device`, and blocks until complete.
 *
 * @param ctx Context used for submit-and-wait.
 * @param staging Host-visible staging buffer large enough for `bytes`.
 * @param device Device-local destination buffer.
 * @param bytes Host bytes to upload (must fit in both buffers).
 */
[[nodiscard]] auto
  upload_to_device(context &ctx,
    owned::gpu_buffer &staging,
    owned::gpu_buffer const &device,
    std::span<std::byte const> bytes)
    -> status;

/**
 * Copies `device` into host-visible `staging`, waits, then writes bytes into `out`.
 *
 * @param ctx Context used for submit-and-wait.
 * @param staging Host-visible staging buffer large enough for `out`.
 * @param device Device-local source buffer.
 * @param out Host destination (must fit in both buffers).
 */
[[nodiscard]] auto
  download_to_host(context &ctx,
    owned::gpu_buffer &staging,
    owned::gpu_buffer const &device,
    std::span<std::byte> out) -> status;

}// namespace vkexec

#endif// VKEXEC_COPY_HPP
