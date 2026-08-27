#include "load_gltf_mesh.hpp"

#include <utility>
#include <vkexec/error.hpp>
#include <vkexec_graphics/mesh.hpp>

#include <tiny_gltf_v3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <iterator>
#include <numbers>
#include <span>
#include <string>
#include <vector>

namespace vkexec::examples {
namespace {

  constexpr int k_gltf_component_float = TG3_COMPONENT_TYPE_FLOAT;
  constexpr int k_gltf_component_unsigned_int = TG3_COMPONENT_TYPE_UNSIGNED_INT;
  constexpr int k_gltf_type_vec3 = TG3_TYPE_VEC3;
  constexpr int k_gltf_type_scalar = TG3_TYPE_SCALAR;
  constexpr int k_gltf_mode_triangles = TG3_MODE_TRIANGLES;
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
  constexpr float k_ambient = 0.15F;
  constexpr float k_fit_extent = 1.6F;
  constexpr float k_center_scale = 0.5F;
  constexpr float k_unit_scale = 1.0F;
  constexpr float k_quaternion_scale = 2.0F;
  constexpr std::size_t k_translation_col = 3;
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

  auto matrix_from_gltf(std::span<double const, k_mat4_element_count> source) -> mat4
  {
    mat4 matrix{};
    for (std::size_t index = 0; index < k_mat4_element_count; ++index) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
      matrix.cols.at(index) = static_cast<float>(source[index]);
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

  auto vec3_from_gltf(double const *source, vec3 fallback) -> vec3
  {
    if (source == nullptr) { return fallback; }
    return vec3{
      .x = static_cast<float>(source[0]),// NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      .y = static_cast<float>(source[1]),// NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      .z = static_cast<float>(source[2]),// NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    };
  }

  auto matrix_from_trs(tg3_node const &node) -> mat4
  {
    auto const translation = vec3_from_gltf(static_cast<double const *>(node.translation), vec3{});
    auto const scale = vec3_from_gltf(
      static_cast<double const *>(node.scale), vec3{ .x = k_unit_scale, .y = k_unit_scale, .z = k_unit_scale });
    auto rotation_x = static_cast<float>(node.rotation[0]);
    auto rotation_y = static_cast<float>(node.rotation[1]);
    auto rotation_z = static_cast<float>(node.rotation[2]);
    auto rotation_w = static_cast<float>(node.rotation[3]);
    auto const translation_matrix = matrix_from_translation(translation);
    auto const rotation_matrix = matrix_from_quaternion(rotation_x, rotation_y, rotation_z, rotation_w);
    auto const scale_matrix = matrix_from_scale(scale);
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

  auto transform_normal(mat4 const &matrix, vec3 normal) -> vec3
  {
    return vec3{
      .x = (mat_at(matrix, k_axis_x, k_axis_x) * normal.x) + (mat_at(matrix, k_axis_y, k_axis_x) * normal.y)
           + (mat_at(matrix, k_axis_z, k_axis_x) * normal.z),
      .y = (mat_at(matrix, k_axis_x, k_axis_y) * normal.x) + (mat_at(matrix, k_axis_y, k_axis_y) * normal.y)
           + (mat_at(matrix, k_axis_z, k_axis_y) * normal.z),
      .z = (mat_at(matrix, k_axis_x, k_axis_z) * normal.x) + (mat_at(matrix, k_axis_y, k_axis_z) * normal.y)
           + (mat_at(matrix, k_axis_z, k_axis_z) * normal.z),
    };
  }

  auto normalize_vec3(vec3 value) -> vec3
  {
    float const length = std::sqrt((value.x * value.x) + (value.y * value.y) + (value.z * value.z));
    if (length <= 0.0F) { return value; }
    return vec3{ .x = value.x / length, .y = value.y / length, .z = value.z / length };
  }

  auto shade_vertex(vec3 normal, std::array<float, k_mesh_vertex_components> const &base_color)
    -> std::array<float, k_mesh_vertex_components>
  {
    constexpr auto k_inv_sqrt3 = std::numbers::inv_sqrt3_v<float>;
    float const n_dot_l =
      std::max(0.0F, (normal.x * k_inv_sqrt3) + (normal.y * k_inv_sqrt3) + (normal.z * k_inv_sqrt3));
    float const intensity = k_ambient + ((k_unit_scale - k_ambient) * n_dot_l);
    return {
      base_color.at(0) * intensity,
      base_color.at(1) * intensity,
      base_color.at(2) * intensity,
    };
  }

  auto node_local_matrix(tg3_node const &node) -> mat4
  {
    if (node.has_matrix != 0) { return matrix_from_gltf(std::span<double const, k_mat4_element_count>(node.matrix)); }
    return matrix_from_trs(node);
  }

  auto material_color(tg3_model const &model, std::int32_t material_index)
    -> std::array<float, k_mesh_vertex_components>
  {
    std::array<float, k_mesh_vertex_components> color{
      k_default_material_color,
      k_default_material_color,
      k_default_material_color,
    };
    if (material_index < 0) { return color; }
    if (std::cmp_greater_equal(material_index, model.materials_count)) { return color; }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    tg3_material const &material = model.materials[static_cast<std::size_t>(material_index)];
    color.at(0) = static_cast<float>(material.pbr_metallic_roughness.base_color_factor[0]);
    color.at(1) = static_cast<float>(material.pbr_metallic_roughness.base_color_factor[1]);
    color.at(2) = static_cast<float>(material.pbr_metallic_roughness.base_color_factor[2]);
    return color;
  }

  auto accessor_at(tg3_model const &model, std::int32_t accessor_index) -> vkexec::result<tg3_accessor const *>
  {
    if (accessor_index < 0) { return make_error(errc::out_of_range, "gltf accessor index out of range"); }
    if (std::cmp_greater_equal(accessor_index, model.accessors_count)) {
      return make_error(errc::out_of_range, "gltf accessor index out of range");
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return &model.accessors[static_cast<std::size_t>(accessor_index)];
  }

  auto buffer_view_at(tg3_model const &model, std::int32_t buffer_view_index) -> vkexec::result<tg3_buffer_view const *>
  {
    if (buffer_view_index < 0) { return make_error(errc::out_of_range, "gltf buffer view index out of range"); }
    if (std::cmp_greater_equal(buffer_view_index, model.buffer_views_count)) {
      return make_error(errc::out_of_range, "gltf buffer view index out of range");
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return &model.buffer_views[static_cast<std::size_t>(buffer_view_index)];
  }

  auto buffer_at(tg3_model const &model, std::int32_t buffer_index) -> vkexec::result<tg3_buffer const *>
  {
    if (buffer_index < 0) { return make_error(errc::out_of_range, "gltf buffer index out of range"); }
    if (std::cmp_greater_equal(buffer_index, model.buffers_count)) {
      return make_error(errc::out_of_range, "gltf buffer index out of range");
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return &model.buffers[static_cast<std::size_t>(buffer_index)];
  }

  auto accessor_bytes(tg3_model const &model, std::int32_t accessor_index)
    -> vkexec::result<std::span<std::uint8_t const>>
  {
    if (accessor_index < 0 || std::cmp_greater_equal(accessor_index, model.accessors_count)) {
      return make_error(errc::out_of_range, "gltf accessor index out of range");
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    tg3_accessor const &accessor = model.accessors[static_cast<std::size_t>(accessor_index)];
    if (accessor.buffer_view < 0) { return make_error(errc::parse_error, "gltf accessor missing buffer view"); }
    if (std::cmp_greater_equal(accessor.buffer_view, model.buffer_views_count)) {
      return make_error(errc::out_of_range, "gltf buffer view index out of range");
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    tg3_buffer_view const &view = model.buffer_views[static_cast<std::size_t>(accessor.buffer_view)];
    if (view.buffer < 0) { return make_error(errc::parse_error, "gltf buffer view missing buffer"); }
    if (std::cmp_greater_equal(view.buffer, model.buffers_count)) {
      return make_error(errc::out_of_range, "gltf buffer index out of range");
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    tg3_buffer const &buffer = model.buffers[static_cast<std::size_t>(view.buffer)];
    std::size_t const offset =
      static_cast<std::size_t>(view.byte_offset) + static_cast<std::size_t>(accessor.byte_offset);
    int const component_size = tg3_component_size(accessor.component_type);
    int const component_count = tg3_num_components(accessor.type);
    if (component_size < 0 || component_count < 0) {
      return make_error(errc::parse_error, "gltf accessor has invalid component layout");
    }
    std::size_t const byte_length = static_cast<std::size_t>(accessor.count) * static_cast<std::size_t>(component_size)
                                    * static_cast<std::size_t>(component_count);
    if (offset + byte_length > buffer.data.count) {
      return make_error(errc::parse_error, "gltf accessor exceeds buffer bounds");
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return std::span<std::uint8_t const>{ buffer.data.data + offset, byte_length };
  }

  auto read_vec3_positions(tg3_model const &model, std::int32_t accessor_index) -> vkexec::result<std::vector<vec3>>
  {
    auto accessor_result = accessor_at(model, accessor_index);
    if (!accessor_result) { return accessor_result.error(); }
    tg3_accessor const &accessor = *accessor_result.value();

    if (accessor.type != k_gltf_type_vec3 || accessor.component_type != k_gltf_component_float) {
      return make_error(errc::parse_error, "gltf POSITION accessor must be VEC3/float");
    }
    if (accessor.buffer_view < 0) {
      return make_error(errc::parse_error, "gltf POSITION accessor missing buffer view");
    }
    auto view_result = buffer_view_at(model, accessor.buffer_view);
    if (!view_result) { return view_result.error(); }
    tg3_buffer_view const &view = *view_result.value();

    if (view.buffer < 0) { return make_error(errc::parse_error, "gltf POSITION buffer view missing buffer"); }
    auto buffer_result = buffer_at(model, view.buffer);
    if (!buffer_result) { return buffer_result.error(); }
    tg3_buffer const &buffer = *buffer_result.value();

    int const stride = tg3_accessor_byte_stride(&accessor, &view);
    if (stride <= 0) { return make_error(errc::parse_error, "gltf POSITION accessor has invalid byte stride"); }

    std::span<std::uint8_t const> const bytes(buffer.data.data, buffer.data.count);
    std::size_t const base_offset =
      static_cast<std::size_t>(view.byte_offset) + static_cast<std::size_t>(accessor.byte_offset);
    std::vector<vec3> positions(static_cast<std::size_t>(accessor.count));
    for (std::size_t index = 0; index < positions.size(); ++index) {
      std::array<float, k_mesh_vertex_components> values{};
      std::size_t const byte_offset = base_offset + (index * static_cast<std::size_t>(stride));
      if (byte_offset + sizeof(values) > bytes.size()) {
        return make_error(errc::parse_error, "gltf POSITION accessor exceeds buffer bounds");
      }
      std::memcpy(values.data(), bytes.subspan(byte_offset, sizeof(values)).data(), sizeof(values));
      positions.at(index) = vec3{ .x = values.at(0), .y = values.at(1), .z = values.at(2) };
    }
    return positions;
  }

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  auto read_vec3_normals(tg3_model const &model, std::int32_t accessor_index, std::size_t expected_count)
    -> vkexec::result<std::vector<vec3>>
  {
    auto accessor_result = accessor_at(model, accessor_index);
    if (!accessor_result) { return accessor_result.error(); }
    tg3_accessor const &accessor = *accessor_result.value();

    if (accessor.type != k_gltf_type_vec3 || accessor.component_type != k_gltf_component_float) {
      return make_error(errc::parse_error, "gltf NORMAL accessor must be VEC3/float");
    }
    if (accessor.buffer_view < 0) { return make_error(errc::parse_error, "gltf NORMAL accessor missing buffer view"); }
    auto view_result = buffer_view_at(model, accessor.buffer_view);
    if (!view_result) { return view_result.error(); }
    tg3_buffer_view const &view = *view_result.value();

    if (view.buffer < 0) { return make_error(errc::parse_error, "gltf NORMAL buffer view missing buffer"); }
    auto buffer_result = buffer_at(model, view.buffer);
    if (!buffer_result) { return buffer_result.error(); }
    tg3_buffer const &buffer = *buffer_result.value();

    int const stride = tg3_accessor_byte_stride(&accessor, &view);
    if (stride <= 0) { return make_error(errc::parse_error, "gltf NORMAL accessor has invalid byte stride"); }

    std::span<std::uint8_t const> const bytes(buffer.data.data, buffer.data.count);
    std::size_t const base_offset =
      static_cast<std::size_t>(view.byte_offset) + static_cast<std::size_t>(accessor.byte_offset);
    std::size_t const count = std::min(static_cast<std::size_t>(accessor.count), expected_count);
    std::vector<vec3> normals(count);
    for (std::size_t index = 0; index < count; ++index) {
      std::array<float, k_mesh_vertex_components> values{};
      std::size_t const byte_offset = base_offset + (index * static_cast<std::size_t>(stride));
      if (byte_offset + sizeof(values) > bytes.size()) {
        return make_error(errc::parse_error, "gltf NORMAL accessor exceeds buffer bounds");
      }
      std::memcpy(values.data(), bytes.subspan(byte_offset, sizeof(values)).data(), sizeof(values));
      normals.at(index) = vec3{ .x = values.at(0), .y = values.at(1), .z = values.at(2) };
    }
    return normals;
  }

  auto read_indices(tg3_model const &model, std::int32_t accessor_index) -> vkexec::result<std::vector<std::uint32_t>>
  {
    auto accessor_result = accessor_at(model, accessor_index);
    if (!accessor_result) { return accessor_result.error(); }
    tg3_accessor const &accessor = *accessor_result.value();

    if (accessor.type != k_gltf_type_scalar || accessor.component_type != k_gltf_component_unsigned_int) {
      return make_error(errc::parse_error, "gltf indices accessor must be SCALAR/unsigned int");
    }
    auto bytes_result = accessor_bytes(model, accessor_index);
    if (!bytes_result) { return bytes_result.error(); }
    std::span<std::uint8_t const> const bytes = bytes_result.value();

    std::vector<std::uint32_t> indices(static_cast<std::size_t>(accessor.count));
    std::memcpy(indices.data(), bytes.data(), bytes.size());
    return indices;
  }

  auto find_primitive_attribute(tg3_primitive const &primitive, char const *name) -> std::int32_t
  {
    for (std::uint32_t index = 0; index < primitive.attributes_count; ++index) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      tg3_str_int_pair const &attribute = primitive.attributes[index];
      if (tg3_str_equals_cstr(attribute.key, name) != 0) { return attribute.value; }
    }
    return TG3_INDEX_NONE;
  }

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  auto append_primitive(tg3_model const &model,
    tg3_primitive const &primitive,
    mat4 const &world,
    std::vector<mesh_vertex> &vertices,
    std::vector<std::uint32_t> &indices) -> vkexec::status
  {
    int const mode = primitive.mode < 0 ? k_gltf_mode_triangles : primitive.mode;
    if (mode != k_gltf_mode_triangles) { return {}; }

    std::int32_t const position_accessor = find_primitive_attribute(primitive, "POSITION");
    if (position_accessor < 0) { return make_error(errc::parse_error, "gltf primitive missing POSITION"); }
    if (primitive.indices < 0) { return make_error(errc::parse_error, "gltf primitive missing indices"); }

    auto local_positions_result = read_vec3_positions(model, position_accessor);
    if (!local_positions_result) { return local_positions_result.error(); }
    std::vector<vec3> const &local_positions = local_positions_result.value();

    auto local_indices_result = read_indices(model, primitive.indices);
    if (!local_indices_result) { return local_indices_result.error(); }
    std::vector<std::uint32_t> const &local_indices = local_indices_result.value();

    std::array<float, k_mesh_vertex_components> const base_color = material_color(model, primitive.material);

    std::int32_t const normal_accessor = find_primitive_attribute(primitive, "NORMAL");
    std::vector<vec3> local_normals;
    if (normal_accessor >= 0) {
      auto local_normals_result = read_vec3_normals(model, normal_accessor, local_positions.size());
      if (!local_normals_result) { return local_normals_result.error(); }
      local_normals = std::move(local_normals_result.value());
    }
    bool const has_normals = local_normals.size() == local_positions.size();

    auto const base_vertex = static_cast<std::uint32_t>(vertices.size());
    vertices.reserve(vertices.size() + local_positions.size());
    for (std::size_t index = 0; index < local_positions.size(); ++index) {
      vec3 const world_position = transform_point(world, local_positions.at(index));
      std::array<float, k_mesh_vertex_components> color = base_color;
      if (has_normals) {
        vec3 const world_normal = normalize_vec3(transform_normal(world, local_normals.at(index)));
        color = shade_vertex(world_normal, base_color);
      }
      vertices.push_back(mesh_vertex{
        .position = { world_position.x, world_position.y, world_position.z },
        .color = color,
      });
    }

    indices.reserve(indices.size() + local_indices.size());
    std::ranges::transform(local_indices,
      std::back_inserter(indices),
      [base_vertex](std::uint32_t local_index) -> std::uint32_t { return base_vertex + local_index; });
    return {};
  }

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  auto append_mesh(tg3_model const &model,
    std::int32_t mesh_index,
    mat4 const &world,
    std::vector<mesh_vertex> &vertices,
    std::vector<std::uint32_t> &indices) -> vkexec::status
  {
    if (mesh_index < 0) { return {}; }
    if (std::cmp_greater_equal(mesh_index, model.meshes_count)) { return {}; }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    tg3_mesh const &mesh = model.meshes[static_cast<std::size_t>(mesh_index)];
    for (std::uint32_t primitive_index = 0; primitive_index < mesh.primitives_count; ++primitive_index) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      if (auto status = append_primitive(model, mesh.primitives[primitive_index], world, vertices, indices); !status) {
        return status.error();
      }
    }
    return {};
  }

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  auto traverse_nodes(tg3_model const &model,
    std::int32_t node_index,
    mat4 const &parent,
    std::vector<mesh_vertex> &vertices,
    std::vector<std::uint32_t> &indices) -> vkexec::status
  {
    if (node_index < 0) { return {}; }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    if (std::cmp_greater_equal(node_index, model.nodes_count)) { return {}; }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    tg3_node const &node = model.nodes[static_cast<std::size_t>(node_index)];
    mat4 const world = multiply(parent, node_local_matrix(node));
    if (auto status = append_mesh(model, node.mesh, world, vertices, indices); !status) { return status.error(); }
    for (std::uint32_t child_index = 0; child_index < node.children_count; ++child_index) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      if (auto status = traverse_nodes(model, node.children[child_index], world, vertices, indices); !status) {
        return status.error();
      }
    }
    return {};
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
      vertex.position.at(1) = -((vertex.position.at(1) - center_y) * scale);
      vertex.position.at(2) = (vertex.position.at(2) - center_z) * scale;
    }
  }

  auto format_tg3_errors(tg3_error_stack const *errors) -> std::string
  {
    std::string message;
    std::uint32_t const count = tg3_errors_count(errors);
    for (std::uint32_t index = 0; index < count; ++index) {
      tg3_error_entry const *entry = tg3_errors_get(errors, index);
      if (entry == nullptr) { continue; }
      if (!message.empty()) { message += "; "; }
      if (entry->message != nullptr) { message += entry->message; }
    }
    if (message.empty()) { message = "unknown tinygltf error"; }
    return message;
  }

}// namespace

auto load_gltf_mesh(std::string const &path) -> vkexec::result<gltf_mesh_data>
{
  tinygltf3::Model model;
  tinygltf3::ErrorStack errors;
  tg3_parse_options options{};
  tg3_parse_options_init(&options);

  tg3_error_code const parse_error =
    tg3_parse_file(model.get(), errors.get(), path.c_str(), static_cast<std::uint32_t>(path.size()), &options);
  if (parse_error != TG3_OK || errors.has_error()) {
    return make_error(
      errc::io_error, std::format("tinygltf failed to load {}: {}", path, format_tg3_errors(errors.get())));
  }

  tg3_model const &gltf = *model.get();
  gltf_mesh_data mesh_data{};
  mat4 const identity = identity_matrix();
  if (gltf.scenes_count == 0) { return make_error(errc::parse_error, "gltf file has no scenes"); }
  int const scene_index = gltf.default_scene >= 0 ? gltf.default_scene : 0;
  if (std::cmp_greater_equal(scene_index, gltf.scenes_count)) {
    return make_error(errc::parse_error, "gltf default scene index out of range");
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  tg3_scene const &scene = gltf.scenes[static_cast<std::size_t>(scene_index)];
  for (std::uint32_t node_index = 0; node_index < scene.nodes_count; ++node_index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    if (auto status = traverse_nodes(gltf, scene.nodes[node_index], identity, mesh_data.vertices, mesh_data.indices);
      !status) {
      return status.error();
    }
  }

  if (mesh_data.vertices.empty() || mesh_data.indices.empty()) {
    return make_error(errc::empty_result, "gltf file contained no triangle geometry");
  }
  fit_mesh_to_clip_space(mesh_data.vertices);
  return mesh_data;
}

}// namespace vkexec::examples
