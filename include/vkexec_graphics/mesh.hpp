#ifndef VKEXEC_GRAPHICS_MESH_HPP
#define VKEXEC_GRAPHICS_MESH_HPP

//! \file
//! Allocator-neutral mesh buffer facts for graphics recording.

#include <vkexec_graphics/graphics_pipeline_resources.hpp>
#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace vkexec {

constexpr std::size_t k_mesh_vertex_components = 3;

struct mesh_vertex
{
  std::array<float, k_mesh_vertex_components> position{};
  std::array<float, k_mesh_vertex_components> color{};
  std::array<float, k_mesh_vertex_components> normal{};
  std::array<float, 2> texcoord{};
};

namespace handles {

  struct mesh
  {
    VkBuffer vertex_buffer{ VK_NULL_HANDLE };
    VkBuffer index_buffer{ VK_NULL_HANDLE };
    std::uint32_t vertex_count{ 0 };
    std::uint32_t index_count{ 0 };
  };

}// namespace handles

namespace graphics {
  using mesh = handles::mesh;
}// namespace graphics

[[nodiscard]] inline auto make_mesh_draw(handles::mesh const &buffers) noexcept -> mesh_draw
{
  return mesh_draw{
    .vertex_buffer = buffers.vertex_buffer,
    .index_buffer = buffers.index_buffer,
    .index_count = buffers.index_count,
  };
}

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_MESH_HPP
