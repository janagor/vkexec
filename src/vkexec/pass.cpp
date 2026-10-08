#include <vkexec/pass.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/barrier_params.hpp>

#include <vkexec/detail/descriptor_backend.hpp>
#include <vkexec/detail/record_with_binding.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace vkexec {

namespace detail {

  auto record_sync(context &facade,
    VkCommandBuffer cmd,
    std::vector<image_barrier_params> const &images,
    std::vector<buffer_barrier_params> const &buffers) -> status
  {
    for (auto const &barrier : images) { VKEXEC_TRY(image_barrier(facade, cmd, barrier)); }
    for (auto const &barrier : buffers) { VKEXEC_TRY(buffer_barrier(facade, cmd, barrier)); }
    return {};
  }

}// namespace detail

namespace {

  [[nodiscard]] auto binding_index(handles::compute_pipeline const &pipe, std::uint32_t slot)
    -> std::optional<std::size_t>
  {
    if (pipe.binding_slots.empty()) { return slot; }
    auto const found = std::ranges::find(pipe.binding_slots, slot);
    if (found == pipe.binding_slots.end()) { return std::nullopt; }
    return static_cast<std::size_t>(found - pipe.binding_slots.begin());
  }

  [[nodiscard]] auto resource_access_for(buffer_access access) noexcept -> resource_access
  {
    switch (access) {
    case buffer_access::readonly:
      return resource_access::read;
    case buffer_access::writeonly:
      return resource_access::write;
    case buffer_access::readwrite:
      return resource_access::read_write;
    }
    return resource_access::read_write;
  }

  auto append_resource_use(compute_bind &bind, resource_ref const &resource, buffer_access binding_access) -> void
  {
    if (resource.kind == resource_kind::sampler) { return; }
    if (resource.kind == resource_kind::sampled_image && binding_access != buffer_access::readonly) {
      bind.complete_resource_metadata = false;
      return;
    }
    if (resource.kind == resource_kind::storage_buffer) {
      bind.buffers.push_back(buffer_use{ .buffer = resource.buffer,
        .size = resource.byte_size,
        .usage = buffer_usage::storage_compute,
        .access = resource_access_for(binding_access) });
      return;
    }
    if (resource.image == VK_NULL_HANDLE || resource.image_range.aspectMask == 0) {
      bind.complete_resource_metadata = false;
      return;
    }
    bind.images.push_back(image_use{ .image = resource.image,
      .range = resource.image_range,
      .usage =
        resource.kind == resource_kind::sampled_image ? image_usage::sampled_compute : image_usage::storage_compute,
      .access =
        resource.kind == resource_kind::sampled_image ? resource_access::read : resource_access_for(binding_access),
      .initial_layout = resource.initial_layout,
      .layout = resource.image_layout });
  }

}// namespace

auto bind_compute(handles::compute_pipeline const &pipe, VkDescriptorSet set) -> compute_bind
{
  return compute_bind{ .pipeline = pipe.pipeline,
    .layout = pipe.pipeline_layout,
    .set = set,
    .images = {},
    .buffers = {},
    .resource_metadata = false,
    .complete_resource_metadata = true };
}

auto bind_compute(handles::compute_pipeline const &pipe, VkDescriptorSet set, resource_table const &table)
  -> compute_bind
{
  auto bind = bind_compute(pipe, set);
  bind.resource_metadata = true;
  bind.complete_resource_metadata =
    pipe.binding_accesses.size() == table.size()
    && (pipe.binding_kinds.empty() || pipe.binding_kinds.size() == pipe.binding_accesses.size());
  std::vector<bool> seen(pipe.binding_accesses.size());
  for (auto const &entry : table.entries()) {
    auto const index = binding_index(pipe, entry.slot);
    if (!index || *index >= seen.size() || seen.at(*index)) {
      bind.complete_resource_metadata = false;
      continue;
    }
    seen.at(*index) = true;
    if (!pipe.binding_kinds.empty() && *index >= pipe.binding_kinds.size()) {
      bind.complete_resource_metadata = false;
      continue;
    }
    resource_kind const expected_kind =
      pipe.binding_kinds.empty() ? resource_kind::storage_buffer : pipe.binding_kinds.at(*index);
    if (expected_kind != entry.resource.kind) {
      bind.complete_resource_metadata = false;
      continue;
    }
    append_resource_use(bind, entry.resource, pipe.binding_accesses.at(*index));
  }
  if (std::ranges::find(seen, false) != seen.end()) { bind.complete_resource_metadata = false; }
  return bind;
}

auto record_pass(VkCommandBuffer cmd,
  compute_bind const &bind,
  void const *push,
  std::uint32_t push_bytes,
  dispatch groups) -> void
{
  auto const *const bytes = static_cast<std::byte const *>(push);
  std::span<std::byte const> const push_data{ bytes, push_bytes };
  (void)detail::bind_and_push<detail::set_descriptor_backend>(
    nullptr, cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bind, push_data);
  vkCmdDispatch(cmd, groups.x, groups.y, groups.z);
}

auto record_pass(VkCommandBuffer cmd,
  compute_bind const &bind,
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
