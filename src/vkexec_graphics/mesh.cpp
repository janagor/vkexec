#include <vkexec_graphics/mesh.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>

namespace vkexec {
namespace {

  struct mapped_buffer
  {
    VkBuffer buffer{ VK_NULL_HANDLE };
    VmaAllocation allocation{ VK_NULL_HANDLE };
    void *mapped{ nullptr };
  };

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  auto create_host_buffer(context const &ctx, VkDeviceSize bytes, VkBufferUsageFlags usage) -> result<mapped_buffer>
  {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = bytes;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    mapped_buffer created{};
    VmaAllocationInfo mapped_info{};
    if (VkResult const result = vmaCreateBuffer(
          ctx.allocator(), &buffer_info, &alloc_info, &created.buffer, &created.allocation, &mapped_info);
      result != VK_SUCCESS) {
      return fail(result, "vmaCreateBuffer failed (mesh)");
    }
    created.mapped = mapped_info.pMappedData;
    if (created.mapped == nullptr) {
      vmaDestroyBuffer(ctx.allocator(), created.buffer, created.allocation);
      return fail(errc::unsupported, "vmaCreateBuffer did not map host-visible memory");
    }
    return created;
  }

  auto count_as_uint32(std::size_t count, char const *what) -> result<std::uint32_t>
  {
    if (count == 0 || count > std::numeric_limits<std::uint32_t>::max()) { return fail(errc::invalid_argument, what); }
    return static_cast<std::uint32_t>(count);
  }

}// namespace

auto destroy_mesh_buffers(context const &ctx, mesh_buffers &buffers) noexcept -> void
{
  if (ctx.allocator() == VK_NULL_HANDLE) {
    buffers = {};
    return;
  }
  if (buffers.index_buffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(ctx.allocator(), buffers.index_buffer, buffers.index_allocation);
  }
  if (buffers.vertex_buffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(ctx.allocator(), buffers.vertex_buffer, buffers.vertex_allocation);
  }
  buffers = {};
}

auto create_mesh_buffers(context &ctx, std::span<mesh_vertex const> vertices, std::span<std::uint32_t const> indices)
  -> result<mesh_buffers>
{
  VKEXEC_TRY_ASSIGN(
    vertices_count, count_as_uint32(vertices.size(), "create_mesh_buffers vertex count must be in (0, UINT32_MAX]"));
  VKEXEC_TRY_ASSIGN(
    indices_count, count_as_uint32(indices.size(), "create_mesh_buffers index count must be in (0, UINT32_MAX]"));

  mesh_buffers owned{};
  owned.vertex_count = vertices_count;
  owned.index_count = indices_count;

  VKEXEC_TRY_ASSIGN(vertex,
    create_host_buffer(ctx, static_cast<VkDeviceSize>(vertices.size_bytes()), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));
  owned.vertex_buffer = vertex.buffer;
  owned.vertex_allocation = vertex.allocation;
  std::memcpy(vertex.mapped, vertices.data(), vertices.size_bytes());

  auto index =
    create_host_buffer(ctx, static_cast<VkDeviceSize>(indices.size_bytes()), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
  if (!index) {
    destroy_mesh_buffers(ctx, owned);
    return fail(index);
  }
  auto const &index_buf = vkexec::expected_get(index);
  owned.index_buffer = index_buf.buffer;
  owned.index_allocation = index_buf.allocation;
  std::memcpy(index_buf.mapped, indices.data(), indices.size_bytes());
  return owned;
}

auto mesh::reset() noexcept -> void
{
  if (ctx_ != nullptr) { destroy_mesh_buffers(*ctx_, buffers_); }
  ctx_ = nullptr;
}

auto factory::mesh(::vkexec::context &ctx, std::span<mesh_vertex const> vertices, std::span<std::uint32_t const> indices)
  -> sender<::vkexec::mesh>
{
  return make_sender<::vkexec::mesh>([&ctx, vertices, indices]() -> result<::vkexec::mesh> {
    VKEXEC_TRY_ASSIGN(owned, create_mesh_buffers(ctx, vertices, indices));
    return ::vkexec::mesh{ &ctx, owned };
  });
}

}// namespace vkexec
