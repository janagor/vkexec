#include <vkexec_extensions/descriptor_heap/pass.hpp>

#include "detail/strategy.hpp"

#include <vkexec/context.hpp>
#include <vkexec/detail/record_with_binding.hpp>
#include <vkexec/detail/viewport.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace vkexec {

auto record_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  dispatch groups) -> status
{
  if (auto bound =
        detail::bind_and_push<detail::heap_descriptor_backend>(&ctx, cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bind, push);
    !bound) {
    return fail(bound);
  }
  vkCmdDispatch(cmd, groups.x, groups.y, groups.z);
  return {};
}

auto record_pass(context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  std::span<std::byte const> push,
  indirect_dispatch groups) -> status
{
  if (auto bound =
        detail::bind_and_push<detail::heap_descriptor_backend>(&ctx, cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bind, push);
    !bound) {
    return fail(bound);
  }
  vkCmdDispatchIndirect(cmd, groups.buffer, groups.offset);
  return {};
}

namespace {

  auto bind_heap_graphics_draw_state(VkCommandBuffer cmd, compute_bind bind, VkExtent2D extent) -> void
  {
    (void)detail::bind_and_push<detail::heap_descriptor_backend>(
      nullptr, cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bind, {});

    detail::set_dynamic_viewport_scissor(cmd, extent);
  }

}// namespace

auto record_draw([[maybe_unused]] context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  VkExtent2D extent,
  std::uint32_t vertex_count) -> void
{
  bind_heap_graphics_draw_state(cmd, bind, extent);
  vkCmdDraw(cmd, vertex_count, 1, 0, 0);
}

auto record_draw_indirect([[maybe_unused]] context const &ctx,
  VkCommandBuffer cmd,
  compute_bind bind,
  VkExtent2D extent,
  VkBuffer buffer,
  VkDeviceSize offset) -> void
{
  bind_heap_graphics_draw_state(cmd, bind, extent);
  vkCmdDrawIndirect(cmd, buffer, offset, 1, sizeof(VkDrawIndirectCommand));
}

}// namespace vkexec
