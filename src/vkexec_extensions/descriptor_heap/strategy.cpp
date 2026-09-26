#include <vkexec_extensions/descriptor_heap/strategy.hpp>

#include "detail/strategy.hpp"

#include <vkexec_extensions/descriptor_heap/descriptor_heap.hpp>
#include <vkexec_extensions/descriptor_heap/push_data.hpp>
#include <vkexec_extensions/descriptor_heap/resource_table.hpp>

#include <vkexec/context.hpp>
#include <vkexec/detail/bind_resources.hpp>
#include <vkexec/detail/descriptor_table_backend.hpp>
#include <vkexec/detail/lower_and_bind_push.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/result.hpp>
#include <vkexec/submit_scope.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace vkexec::detail {

namespace {

  [[nodiscard]] auto descriptor_destination(std::span<std::byte> bytes,
    std::size_t descriptor_size,
    std::size_t descriptor_stride,
    std::uint32_t index) -> result<std::span<std::byte>>
  {
    if (descriptor_size == 0 || descriptor_stride == 0) {
      return fail(errc::invalid_argument, "heap descriptor sizes must be non-zero");
    }
    if (index > std::numeric_limits<std::size_t>::max() / descriptor_stride) {
      return fail(errc::invalid_argument, "heap descriptor index overflows mapped range");
    }
    std::size_t const offset = static_cast<std::size_t>(index) * descriptor_stride;
    if (offset > bytes.size() || descriptor_size > bytes.size() - offset) {
      return fail(errc::invalid_argument, "heap descriptor index is outside mapped range");
    }
    return bytes.subspan(offset, descriptor_size);
  }

  struct lower_positions
  {
    std::size_t resource{ 0 };
    std::size_t image{ 0 };
    std::size_t sampler{ 0 };
  };

  [[nodiscard]] auto lower_entry(context &ctx,
    resource_binding const &entry,
    heap_table_lower_env const &env,
    lower_positions &positions) -> result<std::uint32_t>
  {
    switch (entry.resource.kind) {
    case resource_kind::storage_buffer: {
      std::uint32_t const index = env.indices.subspan(positions.resource).front();
      VKEXEC_TRY_ASSIGN(destination,
        descriptor_destination(env.resource_heap_bytes, env.buffer_descriptor_size, env.descriptor_stride, index));
      VkBufferDeviceAddressInfo address_info{};
      address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
      address_info.buffer = entry.resource.buffer;
      VkDeviceAddress const address = vkGetBufferDeviceAddress(ctx.device(), &address_info);
      VKEXEC_TRY(write_storage_buffer_descriptor(ctx, address, entry.resource.byte_size, destination));
      ++positions.resource;
      return index;
    }
    case resource_kind::storage_image:
    case resource_kind::sampled_image: {
      std::uint32_t const index = env.indices.subspan(positions.resource).front();
      VKEXEC_TRY_ASSIGN(destination,
        descriptor_destination(env.resource_heap_bytes, env.image_descriptor_size, env.descriptor_stride, index));
      VkImageViewCreateInfo const &view_info = env.image_view_infos.subspan(positions.image).front();
      if (entry.resource.kind == resource_kind::storage_image) {
        VKEXEC_TRY(write_storage_image_descriptor(ctx, view_info, entry.resource.image_layout, destination));
      } else {
        VKEXEC_TRY(write_sampled_image_descriptor(ctx, view_info, entry.resource.image_layout, destination));
      }
      ++positions.resource;
      ++positions.image;
      return index;
    }
    case resource_kind::sampler: {
      std::uint32_t const index = env.sampler_indices.subspan(positions.sampler).front();
      VKEXEC_TRY_ASSIGN(destination,
        descriptor_destination(
          env.sampler_heap_bytes, env.sampler_descriptor_size, env.sampler_descriptor_stride, index));
      VkSamplerCreateInfo const &sampler_info = env.sampler_infos.subspan(positions.sampler).front();
      VKEXEC_TRY(write_sampler_descriptor(ctx, sampler_info, destination));
      ++positions.sampler;
      return index;
    }
    }
    return fail(errc::invalid_argument, "resource_table contains an unknown resource kind");
  }

}// namespace

auto heap_descriptor_backend::lower(context &ctx,
  [[maybe_unused]] handles::compute_pipeline const &pipe,
  resource_table const &table,
  lower_env const &env) -> result<bound_type>
{
  auto const sampler_count = static_cast<std::size_t>(std::ranges::count_if(table.entries(),
    [](resource_binding const &entry) -> bool { return entry.resource.kind == resource_kind::sampler; }));
  auto const image_count =
    static_cast<std::size_t>(std::ranges::count_if(table.entries(), [](resource_binding const &entry) -> bool {
      return entry.resource.kind == resource_kind::storage_image || entry.resource.kind == resource_kind::sampled_image;
    }));
  if (env.indices.size() != table.size() - sampler_count || env.sampler_indices.size() != sampler_count) {
    return fail(errc::invalid_argument, "heap descriptor indices must match resource kinds");
  }
  if (env.image_view_infos.size() != image_count || env.sampler_infos.size() != sampler_count) {
    return fail(errc::invalid_argument, "heap descriptor create infos must match resource kinds");
  }

  std::vector<heap_index_binding> lowered;
  lowered.reserve(table.size());
  lower_positions positions{};
  for (resource_binding const &entry : table.entries()) {
    if (std::ranges::any_of(
          lowered, [&entry](heap_index_binding const &existing) -> bool { return existing.slot == entry.slot; })) {
      return fail(errc::invalid_argument, "resource_table contains duplicate logical slots");
    }
    VKEXEC_TRY_ASSIGN(index, lower_entry(ctx, entry, env, positions));
    lowered.push_back(heap_index_binding{ .slot = entry.slot, .index = index });
  }
  return heap_index_map{ std::move(lowered) };
}

auto heap_descriptor_backend::push_bytes(context const *ctx,
  VkCommandBuffer cmd,
  [[maybe_unused]] VkPipelineBindPoint bind_point,
  std::span<std::byte const> bytes) -> status
{
  if (bytes.empty()) { return {}; }
  if (ctx == nullptr) { return fail(errc::invalid_argument, "heap descriptor push requires a context"); }
  return cmd_push_data(*ctx, cmd, bytes);
}

static_assert(descriptor_table_backend<heap_descriptor_backend>);

}// namespace vkexec::detail

namespace vkexec {

namespace detail {
  struct descriptor_heap_bind_resources_step_state : bound_release_state<heap_descriptor_backend>
  {
  };
}// namespace detail

auto bind_resources(descriptor_heap_t /*strategy*/,
  handles::compute_pipeline const &pipe,
  resource_table const &table,
  heap_table_lower_env env,
  std::span<std::byte const> push) -> descriptor_heap_bind_resources_closure
{
  return descriptor_heap_bind_resources_closure{
    .pipe = &pipe,
    .table = table,
    .env = env,
    .push = std::vector<std::byte>(push.begin(), push.end()),
    .state = {},
  };
}

auto detail::record_descriptor_heap_bind_resources_step(context &ctx,
  VkCommandBuffer cmd,
  handles::compute_pipeline const *pipe,
  resource_table const &table,
  heap_table_lower_env env,
  std::span<std::byte const> push,
  std::shared_ptr<descriptor_heap_bind_resources_step_state> &state) -> status
{
  using backend = heap_descriptor_backend;
  if (pipe == nullptr) { return fail(errc::invalid_argument, "bind_resources requires a pipeline"); }
  if (!state) {
    state = std::make_shared<descriptor_heap_bind_resources_step_state>();
  } else {
    state->release();
  }
  auto lowered = lower_and_bind_push<backend>(ctx, cmd, VK_PIPELINE_BIND_POINT_COMPUTE, *pipe, table, env, push);
  if (!lowered) { return fail(lowered); }
  state->ctx = &ctx;
  state->pipe = pipe;
  state->bound.emplace(expected_take(lowered));
  return {};
}

auto detail::release_descriptor_heap_bind_resources_step(
  std::shared_ptr<descriptor_heap_bind_resources_step_state> const &state) -> void
{
  if (state) { state->release(); }
}

auto descriptor_heap_bind_resources_closure::record(context &ctx,
  VkCommandBuffer cmd,
  [[maybe_unused]] detail::pass_cleanup &cleanup) -> status
{
  return detail::record_descriptor_heap_bind_resources_step(
    ctx, cmd, pipe, table, env, std::span<std::byte const>{ push }, state);
}

auto descriptor_heap_bind_resources_closure::after_gpu() const -> void
{ detail::release_descriptor_heap_bind_resources_step(state); }

auto heap_index_map::index_for(std::uint32_t slot) const noexcept -> std::optional<std::uint32_t>
{
  auto const found =
    std::ranges::find_if(entries_, [slot](heap_index_binding const &entry) -> bool { return entry.slot == slot; });
  if (found != entries_.end()) { return found->index; }
  return std::nullopt;
}

}// namespace vkexec
