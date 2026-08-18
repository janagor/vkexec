#include "load_gltf_mesh.hpp"

#include <vkexec_graphics/mesh.hpp>

#define TINYGLTF_NO_EXTERNAL_IMAGE
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace vkexec::examples {
namespace {

  constexpr int k_gltf_component_float = TINYGLTF_COMPONENT_TYPE_FLOAT;
  constexpr int k_gltf_component_unsigned_int = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
  constexpr int k_gltf_type_vec3 = TINYGLTF_TYPE_VEC3;
  constexpr int k_gltf_type_scalar = TINYGLTF_TYPE_SCALAR;
  constexpr int k_gltf_mode_triangles = TINYGLTF_MODE_TRIANGLES;
  constexpr std::size_t k_mat4_element_count = 16;
  constexpr std::size_t k_matrix_stride = 4;
  constexpr std::size_t k_matrix_diag_xx = 0;
  constexpr std::size_t k_matrix_diag_yy = 5;
  constexpr std::size_t k_matrix_diag_zz = 10;
  constexpr std::size_t k_matrix_diag_ww = 15;
  constexpr std::size_t k_axis_x = 0;
  constexpr std::size_t k_axis_y = 1;
  constexpr std::size_t k_axis_z = 2;
  constexpr std::size_t k_axis_w = 3;
  constexpr float k_default_material_color = 0.7F;
  constexpr float k_fit_extent = 1.6F;
  constexpr float k_center_scale = 0.5F;
  constexpr float k_unit_scale = 1.0F;
  constexpr float k_quaternion_scale = 2.0F;
  constexpr std::size_t k_translation_col = 3;
  constexpr std::size_t k_quaternion_component_count = 4;
  constexpr std::size_t k_mat_col0_row1 = 1;
  constexpr std::size_t k_mat_col0_row2 = 2;
  constexpr std::size_t k_mat_col1_row0 = 4;
  constexpr std::size_t k_mat_col1_row2 = 6;
  constexpr std::size_t k_mat_col2_row0 = 8;
  constexpr std::size_t k_mat_col2_row1 = 9;

  struct mat4
  {
    std::array<float, k_mat4_element_count> cols{};
  };

  struct vec3
  {
    float x{ 0.0F };
    float y{ 0.0F };
    float z{ 0.0F };
  };

  auto identity_matrix() -> mat4
  {
    mat4 matrix{};
    matrix.cols.at(k_matrix_diag_xx) = 1.0F;
    matrix.cols.at(k_matrix_diag_yy) = 1.0F;
    matrix.cols.at(k_matrix_diag_zz) = 1.0F;
    matrix.cols.at(k_matrix_diag_ww) = 1.0F;
    return matrix;
  }

  auto matrix_from_gltf(std::vector<double> const &source) -> mat4
  {
    if (source.size() != k_mat4_element_count) { throw std::runtime_error("gltf node matrix must have 16 elements"); }
    mat4 matrix{};
    for (std::size_t index = 0; index < k_mat4_element_count; ++index) {
      matrix.cols.at(index) = static_cast<float>(source.at(index));
    }
    return matrix;
  }

  auto multiply(mat4 const &lhs, mat4 const &rhs) -> mat4
  {
    mat4 out{};
    for (std::size_t col = 0; col < k_matrix_stride; ++col) {
      for (std::size_t row = 0; row < k_matrix_stride; ++row) {
        float sum = 0.0F;
        for (std::size_t k = 0; k < k_matrix_stride; ++k) {
          sum += lhs.cols.at((k * k_matrix_stride) + row) * rhs.cols.at((col * k_matrix_stride) + k);
        }
        out.cols.at((col * k_matrix_stride) + row) = sum;
      }
    }
    return out;
  }

  auto matrix_from_translation(vec3 translation) -> mat4
  {
    mat4 matrix = identity_matrix();
    matrix.cols.at((k_translation_col * k_matrix_stride) + k_axis_x) = translation.x;
    matrix.cols.at((k_translation_col * k_matrix_stride) + k_axis_y) = translation.y;
    matrix.cols.at((k_translation_col * k_matrix_stride) + k_axis_z) = translation.z;
    return matrix;
  }

  auto matrix_from_scale(vec3 scale) -> mat4
  {
    mat4 matrix = identity_matrix();
    matrix.cols.at(k_matrix_diag_xx) = scale.x;
    matrix.cols.at(k_matrix_diag_yy) = scale.y;
    matrix.cols.at(k_matrix_diag_zz) = scale.z;
    return matrix;
  }

  auto matrix_from_quaternion(float coord_x, float coord_y, float coord_z, float coord_w) -> mat4
  {
    float const prod_xx = coord_x * coord_x;
    float const prod_yy = coord_y * coord_y;
    float const prod_zz = coord_z * coord_z;
    float const prod_xy = coord_x * coord_y;
    float const prod_xz = coord_x * coord_z;
    float const prod_yz = coord_y * coord_z;
    float const prod_wx = coord_w * coord_x;
    float const prod_wy = coord_w * coord_y;
    float const prod_wz = coord_w * coord_z;

    mat4 matrix = identity_matrix();
    matrix.cols.at(k_matrix_diag_xx) = k_unit_scale - (k_quaternion_scale * (prod_yy + prod_zz));
    matrix.cols.at(k_matrix_diag_yy) = k_unit_scale - (k_quaternion_scale * (prod_xx + prod_zz));
    matrix.cols.at(k_matrix_diag_zz) = k_unit_scale - (k_quaternion_scale * (prod_xx + prod_yy));
    matrix.cols.at(k_mat_col0_row1) = k_quaternion_scale * (prod_xy + prod_wz);
    matrix.cols.at(k_mat_col1_row0) = k_quaternion_scale * (prod_xy - prod_wz);
    matrix.cols.at(k_mat_col0_row2) = k_quaternion_scale * (prod_xz - prod_wy);
    matrix.cols.at(k_mat_col2_row0) = k_quaternion_scale * (prod_xz + prod_wy);
    matrix.cols.at(k_mat_col1_row2) = k_quaternion_scale * (prod_yz + prod_wx);
    matrix.cols.at(k_mat_col2_row1) = k_quaternion_scale * (prod_yz - prod_wx);
    return matrix;
  }

  auto vec3_from_gltf(std::vector<double> const &source, vec3 fallback) -> vec3
  {
    if (source.size() != k_mesh_vertex_components) { return fallback; }
    return vec3{
      .x = static_cast<float>(source.at(0)),
      .y = static_cast<float>(source.at(1)),
      .z = static_cast<float>(source.at(2)),
    };
  }

  auto matrix_from_trs(tinygltf::Node const &node) -> mat4
  {
    vec3 const translation = vec3_from_gltf(node.translation, vec3{});
    vec3 const scale = vec3_from_gltf(node.scale, vec3{ .x = k_unit_scale, .y = k_unit_scale, .z = k_unit_scale });
    float rotation_x = 0.0F;
    float rotation_y = 0.0F;
    float rotation_z = 0.0F;
    float rotation_w = k_unit_scale;
    if (node.rotation.size() == k_quaternion_component_count) {
      rotation_x = static_cast<float>(node.rotation.at(0));
      rotation_y = static_cast<float>(node.rotation.at(1));
      rotation_z = static_cast<float>(node.rotation.at(2));
      rotation_w = static_cast<float>(node.rotation.at(3));
    }
    mat4 const translation_matrix = matrix_from_translation(translation);
    mat4 const rotation_matrix = matrix_from_quaternion(rotation_x, rotation_y, rotation_z, rotation_w);
    mat4 const scale_matrix = matrix_from_scale(scale);
    return multiply(multiply(translation_matrix, rotation_matrix), scale_matrix);
  }

  auto mat_at(mat4 const &matrix, std::size_t column, std::size_t row) -> float
  { return matrix.cols.at((column * k_matrix_stride) + row); }

  auto transform_point(mat4 const &matrix, vec3 point) -> vec3
  {
    return vec3{
      .x = (mat_at(matrix, k_axis_x, k_axis_x) * point.x) + (mat_at(matrix, k_axis_y, k_axis_x) * point.y)
           + (mat_at(matrix, k_axis_z, k_axis_x) * point.z) + mat_at(matrix, k_axis_w, k_axis_x),
      .y = (mat_at(matrix, k_axis_x, k_axis_y) * point.x) + (mat_at(matrix, k_axis_y, k_axis_y) * point.y)
           + (mat_at(matrix, k_axis_z, k_axis_y) * point.z) + mat_at(matrix, k_axis_w, k_axis_y),
      .z = (mat_at(matrix, k_axis_x, k_axis_z) * point.x) + (mat_at(matrix, k_axis_y, k_axis_z) * point.y)
           + (mat_at(matrix, k_axis_z, k_axis_z) * point.z) + mat_at(matrix, k_axis_w, k_axis_z),
    };
  }

  auto node_local_matrix(tinygltf::Node const &node) -> mat4
  {
    if (node.matrix.size() == k_mat4_element_count) { return matrix_from_gltf(node.matrix); }
    return matrix_from_trs(node);
  }

  auto material_color(tinygltf::Model const &model, int material_index) -> std::array<float, k_mesh_vertex_components>
  {
    std::array<float, k_mesh_vertex_components> color{
      k_default_material_color,
      k_default_material_color,
      k_default_material_color,
    };
    if (material_index < 0) { return color; }
    auto const material_count = static_cast<int>(model.materials.size());
    if (material_index >= material_count) { return color; }
    tinygltf::Material const &material = model.materials.at(static_cast<std::size_t>(material_index));
    color.at(0) = static_cast<float>(material.pbrMetallicRoughness.baseColorFactor.at(0));
    color.at(1) = static_cast<float>(material.pbrMetallicRoughness.baseColorFactor.at(1));
    color.at(2) = static_cast<float>(material.pbrMetallicRoughness.baseColorFactor.at(2));
    return color;
  }

  auto accessor_bytes(tinygltf::Model const &model, int accessor_index) -> std::span<unsigned char const>
  {
    if (accessor_index < 0) { throw std::runtime_error("gltf accessor index out of range"); }
    tinygltf::Accessor const &accessor = model.accessors.at(static_cast<std::size_t>(accessor_index));
    if (accessor.bufferView < 0) { throw std::runtime_error("gltf accessor missing buffer view"); }
    tinygltf::BufferView const &view = model.bufferViews.at(static_cast<std::size_t>(accessor.bufferView));
    if (view.buffer < 0) { throw std::runtime_error("gltf buffer view missing buffer"); }
    tinygltf::Buffer const &buffer = model.buffers.at(static_cast<std::size_t>(view.buffer));
    std::size_t const offset =
      static_cast<std::size_t>(view.byteOffset) + static_cast<std::size_t>(accessor.byteOffset);
    std::size_t const byte_length =
      static_cast<std::size_t>(accessor.count)
      * static_cast<std::size_t>(tinygltf::GetComponentSizeInBytes(static_cast<std::uint32_t>(accessor.componentType)));
    if (offset + byte_length > buffer.data.size()) { throw std::runtime_error("gltf accessor exceeds buffer bounds"); }
    return std::span<unsigned char const>(buffer.data).subspan(offset, byte_length);
  }

  auto read_vec3_positions(tinygltf::Model const &model, int accessor_index) -> std::vector<vec3>
  {
    tinygltf::Accessor const &accessor = model.accessors.at(static_cast<std::size_t>(accessor_index));
    if (accessor.type != k_gltf_type_vec3 || accessor.componentType != k_gltf_component_float) {
      throw std::runtime_error("gltf POSITION accessor must be VEC3/float");
    }
    if (accessor.bufferView < 0) { throw std::runtime_error("gltf POSITION accessor missing buffer view"); }
    tinygltf::BufferView const &view = model.bufferViews.at(static_cast<std::size_t>(accessor.bufferView));
    if (view.buffer < 0) { throw std::runtime_error("gltf POSITION buffer view missing buffer"); }
    tinygltf::Buffer const &buffer = model.buffers.at(static_cast<std::size_t>(view.buffer));
    int const stride = accessor.ByteStride(view);
    if (stride <= 0) { throw std::runtime_error("gltf POSITION accessor has invalid byte stride"); }

    std::span<unsigned char const> const bytes(buffer.data);
    std::size_t const base_offset =
      static_cast<std::size_t>(view.byteOffset) + static_cast<std::size_t>(accessor.byteOffset);
    std::vector<vec3> positions(static_cast<std::size_t>(accessor.count));
    for (std::size_t index = 0; index < positions.size(); ++index) {
      std::array<float, k_mesh_vertex_components> values{};
      std::size_t const byte_offset = base_offset + (index * static_cast<std::size_t>(stride));
      if (byte_offset + sizeof(values) > bytes.size()) {
        throw std::runtime_error("gltf POSITION accessor exceeds buffer bounds");
      }
      std::memcpy(values.data(), bytes.subspan(byte_offset, sizeof(values)).data(), sizeof(values));
      positions.at(index) = vec3{ .x = values.at(0), .y = values.at(1), .z = values.at(2) };
    }
    return positions;
  }

  auto read_indices(tinygltf::Model const &model, int accessor_index) -> std::vector<std::uint32_t>
  {
    tinygltf::Accessor const &accessor = model.accessors.at(static_cast<std::size_t>(accessor_index));
    if (accessor.type != k_gltf_type_scalar || accessor.componentType != k_gltf_component_unsigned_int) {
      throw std::runtime_error("gltf indices accessor must be SCALAR/unsigned int");
    }
    std::span<unsigned char const> const bytes = accessor_bytes(model, accessor_index);
    std::vector<std::uint32_t> indices(static_cast<std::size_t>(accessor.count));
    std::memcpy(indices.data(), bytes.data(), bytes.size());
    return indices;
  }

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  auto append_primitive(tinygltf::Model const &model,
    tinygltf::Primitive const &primitive,
    mat4 const &world,
    std::vector<mesh_vertex> &vertices,
    std::vector<std::uint32_t> &indices) -> void
  {
    if (primitive.mode != k_gltf_mode_triangles) { return; }
    auto const position_it = primitive.attributes.find("POSITION");
    if (position_it == primitive.attributes.end()) { throw std::runtime_error("gltf primitive missing POSITION"); }
    if (primitive.indices < 0) { throw std::runtime_error("gltf primitive missing indices"); }

    std::vector<vec3> const local_positions = read_vec3_positions(model, position_it->second);
    std::vector<std::uint32_t> const local_indices = read_indices(model, primitive.indices);
    std::array<float, k_mesh_vertex_components> const color = material_color(model, primitive.material);

    auto const base_vertex = static_cast<std::uint32_t>(vertices.size());
    vertices.reserve(vertices.size() + local_positions.size());
    std::ranges::transform(local_positions, std::back_inserter(vertices), [&](vec3 const &local) -> mesh_vertex {
      vec3 const world_position = transform_point(world, local);
      return mesh_vertex{
        .position = { world_position.x, world_position.y, world_position.z },
        .color = color,
      };
    });

    indices.reserve(indices.size() + local_indices.size());
    std::ranges::transform(local_indices, std::back_inserter(indices), [base_vertex](std::uint32_t local_index)-> std::uint32_t {
      return base_vertex + local_index;
    });
  }

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  auto append_mesh(tinygltf::Model const &model,
    int mesh_index,
    mat4 const &world,
    std::vector<mesh_vertex> &vertices,
    std::vector<std::uint32_t> &indices) -> void
  {
    if (mesh_index < 0) { return; }
    tinygltf::Mesh const &mesh = model.meshes.at(static_cast<std::size_t>(mesh_index));
    for (tinygltf::Primitive const &primitive : mesh.primitives) {
      append_primitive(model, primitive, world, vertices, indices);
    }
  }

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  auto traverse_nodes(tinygltf::Model const &model,
    int node_index,
    mat4 const &parent,
    std::vector<mesh_vertex> &vertices,
    std::vector<std::uint32_t> &indices) -> void
  {
    if (node_index < 0) { return; }
    auto const node_count = static_cast<int>(model.nodes.size());
    if (node_index >= node_count) { return; }
    tinygltf::Node const &node = model.nodes.at(static_cast<std::size_t>(node_index));
    mat4 const world = multiply(parent, node_local_matrix(node));
    append_mesh(model, node.mesh, world, vertices, indices);
    for (int const child : node.children) { traverse_nodes(model, child, world, vertices, indices); }
  }

  auto fit_mesh_to_clip_space(std::vector<mesh_vertex> &vertices) -> void
  {
    if (vertices.empty()) { return; }
    float minimum_x = vertices.front().position.at(0);
    float minimum_y = vertices.front().position.at(1);
    float minimum_z = vertices.front().position.at(2);
    float maximum_x = minimum_x;
    float maximum_y = minimum_y;
    float maximum_z = minimum_z;
    for (mesh_vertex const &vertex : vertices) {
      minimum_x = std::min(minimum_x, vertex.position.at(0));
      minimum_y = std::min(minimum_y, vertex.position.at(1));
      minimum_z = std::min(minimum_z, vertex.position.at(2));
      maximum_x = std::max(maximum_x, vertex.position.at(0));
      maximum_y = std::max(maximum_y, vertex.position.at(1));
      maximum_z = std::max(maximum_z, vertex.position.at(2));
    }
    float const center_x = (minimum_x + maximum_x) * k_center_scale;
    float const center_y = (minimum_y + maximum_y) * k_center_scale;
    float const center_z = (minimum_z + maximum_z) * k_center_scale;
    float const extent_x = maximum_x - minimum_x;
    float const extent_y = maximum_y - minimum_y;
    float const extent_z = maximum_z - minimum_z;
    float const max_extent = std::max({ extent_x, extent_y, extent_z });
    if (max_extent <= 0.0F) { return; }
    float const scale = k_fit_extent / max_extent;
    for (mesh_vertex &vertex : vertices) {
      vertex.position.at(0) = (vertex.position.at(0) - center_x) * scale;
      vertex.position.at(1) = (vertex.position.at(1) - center_y) * scale;
      vertex.position.at(2) = (vertex.position.at(2) - center_z) * scale;
    }
  }

}// namespace

auto load_gltf_mesh(std::string const &path) -> gltf_mesh_data
{
  tinygltf::TinyGLTF loader{};
  tinygltf::Model model{};
  std::string error{};
  std::string warning{};

  bool const loaded = path.ends_with(".glb") ? loader.LoadBinaryFromFile(&model, &error, &warning, path)
                                             : loader.LoadASCIIFromFile(&model, &error, &warning, path);
  if (!loaded) { throw std::runtime_error(std::format("tinygltf failed to load {}: {}", path, error)); }

  gltf_mesh_data mesh_data{};
  mat4 const identity = identity_matrix();
  if (model.scenes.empty()) { throw std::runtime_error("gltf file has no scenes"); }
  int const scene_index = model.defaultScene >= 0 ? model.defaultScene : 0;
  tinygltf::Scene const &scene = model.scenes.at(static_cast<std::size_t>(scene_index));
  for (int const node_index : scene.nodes) {
    traverse_nodes(model, node_index, identity, mesh_data.vertices, mesh_data.indices);
  }

  if (mesh_data.vertices.empty() || mesh_data.indices.empty()) {
    throw std::runtime_error("gltf file contained no triangle geometry");
  }
  fit_mesh_to_clip_space(mesh_data.vertices);
  return mesh_data;
}

}// namespace vkexec::examples
