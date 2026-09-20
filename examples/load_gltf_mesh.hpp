#ifndef VKEXEC_EXAMPLES_LOAD_GLTF_MESH_HPP
#define VKEXEC_EXAMPLES_LOAD_GLTF_MESH_HPP

#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec_graphics/mesh.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace vkexec::examples {

struct gltf_mesh_data
{
  std::vector<mesh_vertex> vertices;
  std::vector<std::uint32_t> indices;
};

/// Load the first triangle mesh from a GLTF/GLB file into vkexec mesh arrays.
[[nodiscard]] auto load_gltf_mesh(std::string const &path) -> vkexec::detail::sync_sender_fn<gltf_mesh_data>;

}// namespace vkexec::examples

#endif// VKEXEC_EXAMPLES_LOAD_GLTF_MESH_HPP
