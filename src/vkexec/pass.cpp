#include <vkexec/pass.hpp>

#include <vkexec/detail/descriptor_backend.hpp>
#include <vkexec/detail/record_with_binding.hpp>
#include <vkexec/pipeline.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace vkexec {

auto bind_compute(handles::compute_pipeline const &pipe, VkDescriptorSet set) -> compute_bind
{ return compute_bind{ .pipeline = pipe.pipeline, .layout = pipe.pipeline_layout, .set = set }; }

auto record_pass(VkCommandBuffer cmd, compute_bind bind, void const *push, std::uint32_t push_bytes, dispatch groups)
  -> void
{
  auto const *const bytes = static_cast<std::byte const *>(push);
  std::span<std::byte const> const push_data{ bytes, push_bytes };
  (void)detail::bind_and_push<detail::set_descriptor_backend>(
    nullptr, cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bind, push_data);
  vkCmdDispatch(cmd, groups.x, groups.y, groups.z);
}

auto record_pass(VkCommandBuffer cmd,
  compute_bind bind,
  void const *push,
  std::uint32_t push_bytes,
  indirect_dispatch groups) -> void
{
  auto const *const bytes = static_cast<std::byte const *>(push);
  std::span<std::byte const> const push_data{ bytes, push_bytes };
  (void)detail::bind_and_push<detail::set_descriptor_backend>(
    nullptr, cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bind, push_data);
  vkCmdDispatchIndirect(cmd, groups.buffer, groups.offset);
}

auto record_pass(VkCommandBuffer cmd,
  handles::compute_pipeline const &pipe,
  VkDescriptorSet set,
  void const *push,
  std::uint32_t push_bytes,
  dispatch groups) -> void
{ record_pass(cmd, bind_compute(pipe, set), push, push_bytes, groups); }

}// namespace vkexec
