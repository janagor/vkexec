#include <vkexec_graphics/mesh.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>

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
      return make_vk_error(result, "vmaCreateBuffer failed (mesh)");
    }
    created.mapped = mapped_info.pMappedData;
    if (created.mapped == nullptr) {
      vmaDestroyBuffer(ctx.allocator(), created.buffer, created.allocation);
      return make_error(errc::unsupported, "vmaCreateBuffer did not map host-visible memory");
    }
    return created;
  }

  auto count_as_uint32(std::size_t count, char const *what) -> result<std::uint32_t>
  {
    if (count == 0 || count > std::numeric_limits<std::uint32_t>::max()) {
      return make_error(errc::invalid_argument, what);
    }
    return static_cast<std::uint32_t>(count);
  }

}// namespace

auto mesh::create(context &ctx, std::span<mesh_vertex const> vertices, std::span<std::uint32_t const> indices)
  -> result<mesh>
{
  mesh created;
  if (auto initialized = created.init(ctx, vertices, indices); !initialized) { return initialized.error(); }
  return created;
}

auto mesh::init(context &ctx, std::span<mesh_vertex const> vertices, std::span<std::uint32_t const> indices) -> status
{
  VKEXEC_LEAF_AUTO(
    vertices_count, count_as_uint32(vertices.size(), "vkexec::mesh vertex count must be in (0, UINT32_MAX]"));
  VKEXEC_LEAF_AUTO(
    indices_count, count_as_uint32(indices.size(), "vkexec::mesh index count must be in (0, UINT32_MAX]"));

  ctx_ = &ctx;
  vertex_count_ = vertices_count;
  index_count_ = indices_count;

  VKEXEC_LEAF_AUTO(vertex,
    create_host_buffer(ctx, static_cast<VkDeviceSize>(vertices.size_bytes()), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));
  vertex_buffer_ = vertex.buffer;
  vertex_allocation_ = vertex.allocation;
  std::memcpy(vertex.mapped, vertices.data(), vertices.size_bytes());

  auto index =
    create_host_buffer(ctx, static_cast<VkDeviceSize>(indices.size_bytes()), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
  if (!index) {
    vmaDestroyBuffer(ctx_->allocator(), vertex_buffer_, vertex_allocation_);
    vertex_buffer_ = VK_NULL_HANDLE;
    vertex_allocation_ = VK_NULL_HANDLE;
    ctx_ = nullptr;
    vertex_count_ = 0;
    index_count_ = 0;
    return index.error();
  }
  auto const &index_buf = detail::leaf_get(index);
  index_buffer_ = index_buf.buffer;
  index_allocation_ = index_buf.allocation;
  std::memcpy(index_buf.mapped, indices.data(), indices.size_bytes());
  return {};
}

mesh::~mesh() { destroy(); }

mesh::mesh(mesh &&other) noexcept
  : ctx_(other.ctx_), vertex_buffer_(other.vertex_buffer_), vertex_allocation_(other.vertex_allocation_),
    index_buffer_(other.index_buffer_), index_allocation_(other.index_allocation_), vertex_count_(other.vertex_count_),
    index_count_(other.index_count_)
{
  other.ctx_ = nullptr;
  other.vertex_buffer_ = VK_NULL_HANDLE;
  other.vertex_allocation_ = VK_NULL_HANDLE;
  other.index_buffer_ = VK_NULL_HANDLE;
  other.index_allocation_ = VK_NULL_HANDLE;
  other.vertex_count_ = 0;
  other.index_count_ = 0;
}

auto mesh::operator=(mesh &&other) noexcept -> mesh &
{
  if (this == &other) { return *this; }
  destroy();
  ctx_ = other.ctx_;
  vertex_buffer_ = other.vertex_buffer_;
  vertex_allocation_ = other.vertex_allocation_;
  index_buffer_ = other.index_buffer_;
  index_allocation_ = other.index_allocation_;
  vertex_count_ = other.vertex_count_;
  index_count_ = other.index_count_;
  other.ctx_ = nullptr;
  other.vertex_buffer_ = VK_NULL_HANDLE;
  other.vertex_allocation_ = VK_NULL_HANDLE;
  other.index_buffer_ = VK_NULL_HANDLE;
  other.index_allocation_ = VK_NULL_HANDLE;
  other.vertex_count_ = 0;
  other.index_count_ = 0;
  return *this;
}

auto mesh::destroy() noexcept -> void
{
  if (ctx_ == nullptr || ctx_->allocator() == VK_NULL_HANDLE) { return; }
  if (index_buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(ctx_->allocator(), index_buffer_, index_allocation_);
    index_buffer_ = VK_NULL_HANDLE;
    index_allocation_ = VK_NULL_HANDLE;
  }
  if (vertex_buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(ctx_->allocator(), vertex_buffer_, vertex_allocation_);
    vertex_buffer_ = VK_NULL_HANDLE;
    vertex_allocation_ = VK_NULL_HANDLE;
  }
}

}// namespace vkexec
