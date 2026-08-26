#ifndef VKEXEC_GRAPHICS_MESH_HPP
#define VKEXEC_GRAPHICS_MESH_HPP

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace vkexec {

constexpr std::size_t k_mesh_vertex_components = 3;

struct mesh_vertex
{
  std::array<float, k_mesh_vertex_components> position{};
  std::array<float, k_mesh_vertex_components> color{};
};

/// Host-visible indexed triangle mesh (vertex + index buffers).
class mesh
{
public:
  [[nodiscard]] static auto
    create(context &ctx, std::span<mesh_vertex const> vertices, std::span<std::uint32_t const> indices) -> result<mesh>;

  ~mesh();

  mesh(mesh const &) = delete;
  auto operator=(mesh const &) -> mesh & = delete;

  mesh(mesh &&other) noexcept;
  auto operator=(mesh &&other) noexcept -> mesh &;

  [[nodiscard]] auto vertex_count() const noexcept -> std::uint32_t { return vertex_count_; }
  [[nodiscard]] auto index_count() const noexcept -> std::uint32_t { return index_count_; }
  [[nodiscard]] auto vk_vertex_buffer() const noexcept -> VkBuffer { return vertex_buffer_; }
  [[nodiscard]] auto vk_index_buffer() const noexcept -> VkBuffer { return index_buffer_; }

private:
  mesh() = default;

  auto init(context &ctx, std::span<mesh_vertex const> vertices, std::span<std::uint32_t const> indices) -> status;
  auto destroy() noexcept -> void;

  context *ctx_{ nullptr };
  VkBuffer vertex_buffer_{ VK_NULL_HANDLE };
  VmaAllocation vertex_allocation_{ VK_NULL_HANDLE };
  VkBuffer index_buffer_{ VK_NULL_HANDLE };
  VmaAllocation index_allocation_{ VK_NULL_HANDLE };
  std::uint32_t vertex_count_{ 0 };
  std::uint32_t index_count_{ 0 };
};

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_MESH_HPP
