#ifndef VKEXEC_GRAPHICS_MESH_HPP
#define VKEXEC_GRAPHICS_MESH_HPP

//! \file
//! Host-visible indexed triangle meshes: borrowable handle bag + owning RAII.

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace vkexec {

//! Number of float components per position or color in `mesh_vertex`.
constexpr std::size_t k_mesh_vertex_components = 3;

//! Interleaved position + color vertex used by mesh-aware graphics pipelines.
struct mesh_vertex
{
  std::array<float, k_mesh_vertex_components> position{};
  std::array<float, k_mesh_vertex_components> color{};
};

/**
 * Borrowable vertex/index buffer handles for one mesh.
 *
 * Fill via `create_mesh_buffers` or an embedder's own allocations. Destroy with
 * `destroy_mesh_buffers` (or owning `mesh`).
 */
struct mesh_buffers
{
  VkBuffer vertex_buffer{ VK_NULL_HANDLE };
  VmaAllocation vertex_allocation{ VK_NULL_HANDLE };
  VkBuffer index_buffer{ VK_NULL_HANDLE };
  VmaAllocation index_allocation{ VK_NULL_HANDLE };
  std::uint32_t vertex_count{ 0 };
  std::uint32_t index_count{ 0 };
};

//! Creates host-visible VMA vertex/index buffers filled from the given spans.
[[nodiscard]] auto create_mesh_buffers(context &ctx,
  std::span<mesh_vertex const> vertices,
  std::span<std::uint32_t const> indices) -> result<mesh_buffers>;

//! Destroys VMA allocations in `buffers` and resets them to null.
auto destroy_mesh_buffers(context const &ctx, mesh_buffers &buffers) noexcept -> void;

//! Builds a `mesh_draw` from borrowed mesh buffer handles.
[[nodiscard]] inline auto make_mesh_draw(mesh_buffers const &buffers) noexcept -> mesh_draw
{
  return mesh_draw{
    .vertex_buffer = buffers.vertex_buffer,
    .index_buffer = buffers.index_buffer,
    .index_count = buffers.index_count,
  };
}

class mesh;

namespace factory {

  /**
   * Creates a mesh from `vertices` and `indices`.
   *
   * @param ctx Context whose VMA allocator owns the buffers.
   * @param vertices Vertex data (copied into the vertex buffer).
   * @param indices Triangle indices (copied into the index buffer).
   */
  [[nodiscard]] auto mesh(::vkexec::context &ctx,
    std::span<mesh_vertex const> vertices,
    std::span<std::uint32_t const> indices) -> sender<::vkexec::mesh>;

}// namespace factory

/**
 * Host-visible indexed triangle mesh (owning wrapper over `mesh_buffers`).
 *
 * @see create_mesh_buffers, graphics_pipeline, draw, factory::mesh
 */
class mesh
{
public:
  ~mesh() { reset(); }

  mesh(mesh const &) = delete;
  auto operator=(mesh const &) -> mesh & = delete;

  mesh(mesh &&other) noexcept
    : ctx_(std::exchange(other.ctx_, nullptr)), buffers_(std::exchange(other.buffers_, mesh_buffers{}))
  {}

  auto operator=(mesh &&other) noexcept -> mesh &
  {
    if (this == &other) { return *this; }
    reset();
    ctx_ = std::exchange(other.ctx_, nullptr);
    buffers_ = std::exchange(other.buffers_, mesh_buffers{});
    return *this;
  }

  //! Borrowed mesh buffer handles.
  [[nodiscard]] auto buffers() const noexcept -> mesh_buffers const & { return buffers_; }
  //! Number of vertices uploaded at creation.
  [[nodiscard]] auto vertex_count() const noexcept -> std::uint32_t { return buffers_.vertex_count; }
  //! Number of indices uploaded at creation.
  [[nodiscard]] auto index_count() const noexcept -> std::uint32_t { return buffers_.index_count; }
  //! Vulkan vertex buffer handle.
  [[nodiscard]] auto vk_vertex_buffer() const noexcept -> VkBuffer { return buffers_.vertex_buffer; }
  //! Vulkan index buffer handle.
  [[nodiscard]] auto vk_index_buffer() const noexcept -> VkBuffer { return buffers_.index_buffer; }

private:
  friend auto factory::mesh(::vkexec::context &ctx,
    std::span<mesh_vertex const> vertices,
    std::span<std::uint32_t const> indices) -> sender<::vkexec::mesh>;

  mesh(context *ctx, mesh_buffers buffers) noexcept : ctx_(ctx), buffers_(buffers) {}

  auto reset() noexcept -> void;

  context *ctx_{ nullptr };
  mesh_buffers buffers_{};
};

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_MESH_HPP
