#ifndef VKEXEC_EXAMPLES_LOAD_GLTF_MESH_HPP
#define VKEXEC_EXAMPLES_LOAD_GLTF_MESH_HPP

#include <vkexec/error.hpp>
#include <vkexec/sender.hpp>
#include <vkexec_graphics/mesh.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace vkexec::examples {

struct gltf_mesh_data
{
  struct primitive_draw
  {
    std::uint32_t first_index{};
    std::uint32_t index_count{};
    std::int32_t material_index{ -1 };
  };

  std::vector<mesh_vertex> vertices;
  std::vector<std::uint32_t> indices;
  std::vector<primitive_draw> draws;
  std::vector<std::string> base_color_textures;
};

namespace detail {

  struct load_gltf_mesh_factory
  {
    std::string path;
    bool fit_to_clip_space{ true };

    [[nodiscard]] auto operator()() const -> vkexec::result<gltf_mesh_data>;
  };

}// namespace detail

/// Load triangle meshes from a GLTF/GLB scene into vkexec mesh arrays.
[[nodiscard]] inline auto load_gltf_mesh(std::string const &path, bool fit_to_clip_space = true)
{ return vkexec::make_sender(detail::load_gltf_mesh_factory{ .path = path, .fit_to_clip_space = fit_to_clip_space }); }

}// namespace vkexec::examples

#endif// VKEXEC_EXAMPLES_LOAD_GLTF_MESH_HPP
