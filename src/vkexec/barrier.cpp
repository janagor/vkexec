#include <vkexec/barrier.hpp>

#include <vkexec/context.hpp>
#include <vkexec/detail/synchronization.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan_core.h>

namespace vkexec {

namespace detail {

  auto legacy_stage_mask(VkPipelineStageFlags2 value) -> result<VkPipelineStageFlags>
  {
    VkPipelineStageFlags result = 0;
    auto take = [&](VkPipelineStageFlags2 from, VkPipelineStageFlags destination) -> void {
      if ((value & from) != 0) {
        result |= destination;
        value &= ~from;
      }
    };
    take(VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT
           | VK_PIPELINE_STAGE_2_RESOLVE_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT);
    take(VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
      VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
    // The remaining core 1.0 stage bits have identical values in both flag types.
    constexpr VkPipelineStageFlags2 k_direct =
      VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT
      | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT
      | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT
      | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
      | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
      | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_HOST_BIT
      | VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT
      | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    result |= static_cast<VkPipelineStageFlags>(value & k_direct);
    value &= ~k_direct;
    if (value != 0) {
      return fail(errc::unsupported, "synchronization2 stage cannot be represented by legacy synchronization");
    }
    return result;
  }

  auto legacy_access_mask(VkAccessFlags2 value) -> result<VkAccessFlags>
  {
    VkAccessFlags result = 0;
    auto take = [&](VkAccessFlags2 from, VkAccessFlags destination) -> void {
      if ((value & from) != 0) {
        result |= destination;
        value &= ~from;
      }
    };
    take(VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
      VK_ACCESS_SHADER_READ_BIT);
    take(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    constexpr VkAccessFlags2 k_direct =
      VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT
      | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT
      | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
      | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
      | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_HOST_READ_BIT
      | VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    result |= static_cast<VkAccessFlags>(value & k_direct);
    value &= ~k_direct;
    if (value != 0) {
      return fail(errc::unsupported, "synchronization2 access cannot be represented by legacy synchronization");
    }
    return result;
  }

  auto validate_scope(VkPipelineStageFlags2 stage, VkAccessFlags2 access) -> status
  {
    if (stage == VK_PIPELINE_STAGE_2_NONE && access != VK_ACCESS_2_NONE) {
      return fail(errc::invalid_argument, "a NONE stage requires an empty access mask");
    }
    return {};
  }

  auto legacy_scope_stage(VkPipelineStageFlags2 stage, VkAccessFlags2 access, bool source)
    -> result<VkPipelineStageFlags>
  {
    VKEXEC_TRY(validate_scope(stage, access));
    if (stage == VK_PIPELINE_STAGE_2_NONE) {
      return source ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }
    return legacy_stage_mask(stage);
  }

}// namespace detail

auto memory_barrier(context &ctx, VkCommandBuffer cmd, memory_barrier_params const &params) -> status
{
  VKEXEC_TRY(detail::validate_scope(params.src_stage, params.src_access));
  VKEXEC_TRY(detail::validate_scope(params.dst_stage, params.dst_access));
  if (detail::synchronization_backend_for(ctx) != detail::synchronization_backend::legacy) {
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = params.src_stage;
    barrier.srcAccessMask = params.src_access;
    barrier.dstStageMask = params.dst_stage;
    barrier.dstAccessMask = params.dst_access;
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    ctx.procs().cmd_pipeline_barrier2(cmd, &dependency);
    return {};
  }
  VKEXEC_TRY_ASSIGN(src_stage, detail::legacy_scope_stage(params.src_stage, params.src_access, true));
  VKEXEC_TRY_ASSIGN(dst_stage, detail::legacy_scope_stage(params.dst_stage, params.dst_access, false));
  VKEXEC_TRY_ASSIGN(src_access, detail::legacy_access_mask(params.src_access));
  VKEXEC_TRY_ASSIGN(dst_access, detail::legacy_access_mask(params.dst_access));
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = src_access;
  barrier.dstAccessMask = dst_access;
  vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  return {};
}

auto buffer_barrier(context &ctx, VkCommandBuffer cmd, buffer_barrier_params const &params) -> status
{
  if (params.buffer == VK_NULL_HANDLE || params.size == 0) {
    return fail(errc::invalid_argument, "buffer_barrier requires a buffer and nonzero size");
  }
  VKEXEC_TRY(detail::validate_scope(params.src_stage, params.src_access));
  VKEXEC_TRY(detail::validate_scope(params.dst_stage, params.dst_access));
  if (detail::synchronization_backend_for(ctx) != detail::synchronization_backend::legacy) {
    VkBufferMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask = params.src_stage;
    barrier.srcAccessMask = params.src_access;
    barrier.dstStageMask = params.dst_stage;
    barrier.dstAccessMask = params.dst_access;
    barrier.srcQueueFamilyIndex = params.src_queue_family;
    barrier.dstQueueFamilyIndex = params.dst_queue_family;
    barrier.buffer = params.buffer;
    barrier.offset = params.offset;
    barrier.size = params.size;
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.bufferMemoryBarrierCount = 1;
    dependency.pBufferMemoryBarriers = &barrier;
    ctx.procs().cmd_pipeline_barrier2(cmd, &dependency);
    return {};
  }
  VKEXEC_TRY_ASSIGN(src_stage, detail::legacy_scope_stage(params.src_stage, params.src_access, true));
  VKEXEC_TRY_ASSIGN(dst_stage, detail::legacy_scope_stage(params.dst_stage, params.dst_access, false));
  VKEXEC_TRY_ASSIGN(src_access, detail::legacy_access_mask(params.src_access));
  VKEXEC_TRY_ASSIGN(dst_access, detail::legacy_access_mask(params.dst_access));
  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask = src_access;
  barrier.dstAccessMask = dst_access;
  barrier.srcQueueFamilyIndex = params.src_queue_family;
  barrier.dstQueueFamilyIndex = params.dst_queue_family;
  barrier.buffer = params.buffer;
  barrier.offset = params.offset;
  barrier.size = params.size;
  vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
  return {};
}

auto image_barrier(context &ctx, VkCommandBuffer cmd, image_barrier_params const &params) -> status
{
  if (params.image == VK_NULL_HANDLE || params.range.aspectMask == 0 || params.range.levelCount == 0
      || params.range.layerCount == 0) {
    return fail(errc::invalid_argument, "image_barrier requires an image and nonempty subresource range");
  }
  VKEXEC_TRY(detail::validate_scope(params.src_stage, params.src_access));
  VKEXEC_TRY(detail::validate_scope(params.dst_stage, params.dst_access));
  if (detail::synchronization_backend_for(ctx) != detail::synchronization_backend::legacy) {
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = params.src_stage;
    barrier.srcAccessMask = params.src_access;
    barrier.dstStageMask = params.dst_stage;
    barrier.dstAccessMask = params.dst_access;
    barrier.oldLayout = params.old_layout;
    barrier.newLayout = params.new_layout;
    barrier.srcQueueFamilyIndex = params.src_queue_family;
    barrier.dstQueueFamilyIndex = params.dst_queue_family;
    barrier.image = params.image;
    barrier.subresourceRange = params.range;
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    ctx.procs().cmd_pipeline_barrier2(cmd, &dependency);
    return {};
  }
  VKEXEC_TRY_ASSIGN(src_stage, detail::legacy_scope_stage(params.src_stage, params.src_access, true));
  VKEXEC_TRY_ASSIGN(dst_stage, detail::legacy_scope_stage(params.dst_stage, params.dst_access, false));
  VKEXEC_TRY_ASSIGN(src_access, detail::legacy_access_mask(params.src_access));
  VKEXEC_TRY_ASSIGN(dst_access, detail::legacy_access_mask(params.dst_access));
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcAccessMask = src_access;
  barrier.dstAccessMask = dst_access;
  barrier.oldLayout = params.old_layout;
  barrier.newLayout = params.new_layout;
  barrier.srcQueueFamilyIndex = params.src_queue_family;
  barrier.dstQueueFamilyIndex = params.dst_queue_family;
  barrier.image = params.image;
  barrier.subresourceRange = params.range;
  vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  return {};
}

namespace barrier {

  auto transfer_to_compute_t::operator()(context &ctx, VkCommandBuffer cmd) const -> status
  {
    return memory_barrier(ctx,
      cmd,
      { .src_stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
        .dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dst_access = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT });
  }

  auto compute_to_compute_t::operator()(context &ctx, VkCommandBuffer cmd) const -> status
  {
    return memory_barrier(ctx,
      cmd,
      { .src_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
        .src_access = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dst_access =
          VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT });
  }

  auto compute_to_graphics_t::operator()(context &ctx, VkCommandBuffer cmd) const -> status
  {
    return memory_barrier(ctx,
      cmd,
      { .src_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dst_stage = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
                     | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .src_access = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dst_access = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT });
  }

  auto graphics_to_compute_t::operator()(context &ctx, VkCommandBuffer cmd) const -> status
  {
    return memory_barrier(ctx,
      cmd,
      { .src_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .src_access = VK_ACCESS_2_SHADER_READ_BIT,
        .dst_access = VK_ACCESS_2_SHADER_WRITE_BIT });
  }

  auto compute_read_t::operator()(context &ctx, VkCommandBuffer cmd) const -> status
  {
    return memory_barrier(ctx,
      cmd,
      { .src_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dst_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .src_access = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dst_access = VK_ACCESS_2_SHADER_READ_BIT });
  }

}// namespace barrier

}// namespace vkexec
