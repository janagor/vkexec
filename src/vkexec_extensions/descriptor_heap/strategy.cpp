#include <vkexec_extensions/descriptor_heap/strategy.hpp>

#include <vkexec_extensions/descriptor_heap/push_data.hpp>
#include <vkexec_extensions/descriptor_heap/descriptor_heap.hpp>
#include <vkexec_extensions/descriptor_heap/resource_table.hpp>

#include <vkexec/context.hpp>
#include <vkexec/detail/descriptor_table_backend.hpp>
#include <vkexec/detail/result.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace vkexec::detail {

namespace {

  [[nodiscard]] auto descriptor_destination(
    heap_table_lower_env const &env, std::uint32_t index) -> result<std::span<std::byte>>
  {
    if (env.buffer_descriptor_size == 0 || env.descriptor_stride == 0) {
      return fail(errc::invalid_argument, "heap descriptor sizes must be non-zero");
    }
    if (index > std::numeric_limits<std::size_t>::max() / env.descriptor_stride) {
      return fail(errc::invalid_argument, "heap descriptor index overflows mapped range");
    }
    std::size_t const offset = static_cast<std::size_t>(index) * env.descriptor_stride;
    if (offset > env.resource_heap_bytes.size()
      || env.buffer_descriptor_size > env.resource_heap_bytes.size() - offset) {
      return fail(errc::invalid_argument, "heap descriptor index is outside mapped range");
    }
    return env.resource_heap_bytes.subspan(offset, env.buffer_descriptor_size);
  }

}// namespace

auto heap_descriptor_backend::lower(
  context &ctx, pipeline_resources const & /*pipe*/, resource_table const &table, lower_env const &env)
  -> result<bound_type>
{
  if (env.indices.size() != table.size()) {
    return fail(errc::invalid_argument, "heap descriptor indices must match resource_table size");
  }

  std::vector<heap_index_binding> lowered;
  lowered.reserve(table.size());
  auto index = env.indices.begin();
  for (resource_binding const &entry : table.entries()) {
    if (std::ranges::any_of(lowered, [&entry](heap_index_binding const &existing) -> bool {
          return existing.slot == entry.slot;
        })) {
      return fail(errc::invalid_argument, "resource_table contains duplicate logical slots");
    }
    VKEXEC_TRY_ASSIGN(destination, descriptor_destination(env, *index));
    VkBufferDeviceAddressInfo address_info{};
    address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    address_info.buffer = entry.resource.buffer;
    VkDeviceAddress const address = vkGetBufferDeviceAddress(ctx.device(), &address_info);
    if (auto written = write_storage_buffer_descriptor(ctx, address, entry.resource.byte_size, destination); !written) {
      return fail(written);
    }
    lowered.push_back(heap_index_binding{ .slot = entry.slot, .index = *index });
    ++index;
  }
  return heap_index_map{ std::move(lowered) };
}

auto heap_descriptor_backend::push_bytes(context const *ctx,
  VkCommandBuffer cmd,
  VkPipelineBindPoint /*bind_point*/,
  std::span<std::byte const> bytes) -> status
{
  if (bytes.empty()) { return {}; }
  if (ctx == nullptr) { return fail(errc::invalid_argument, "heap descriptor push requires a context"); }
  return cmd_push_data(*ctx, cmd, bytes);
}

static_assert(descriptor_table_backend<heap_descriptor_backend>);

}// namespace vkexec::detail

namespace vkexec {

auto heap_index_map::index_for(std::uint32_t slot) const noexcept -> std::optional<std::uint32_t>
{
  auto const found = std::ranges::find_if(entries_, [slot](heap_index_binding const &entry) -> bool {
    return entry.slot == slot;
  });
  if (found != entries_.end()) { return found->index; }
  return std::nullopt;
}

}// namespace vkexec
