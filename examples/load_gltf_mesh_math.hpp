#ifndef VKEXEC_EXAMPLES_LOAD_GLTF_MESH_MATH_HPP
#define VKEXEC_EXAMPLES_LOAD_GLTF_MESH_MATH_HPP

#include "load_gltf_mesh.hpp"

#include <tiny_gltf_v3.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>

namespace vkexec::examples::detail {

constexpr int k_gltf_component_float = TG3_COMPONENT_TYPE_FLOAT;
constexpr int k_gltf_component_unsigned_int = TG3_COMPONENT_TYPE_UNSIGNED_INT;
constexpr int k_gltf_component_unsigned_short = TG3_COMPONENT_TYPE_UNSIGNED_SHORT;
constexpr int k_gltf_component_unsigned_byte = TG3_COMPONENT_TYPE_UNSIGNED_BYTE;
constexpr int k_gltf_type_vec3 = TG3_TYPE_VEC3;
constexpr int k_gltf_type_vec2 = TG3_TYPE_VEC2;
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

inline auto identity_matrix() -> mat4
{
  mat4 matrix{};
  matrix.cols.at(k_matrix_diag_xx) = 1.0F;
  matrix.cols.at(k_matrix_diag_yy) = 1.0F;
  matrix.cols.at(k_matrix_diag_zz) = 1.0F;
  matrix.cols.at(k_matrix_diag_ww) = 1.0F;
  return matrix;
}

inline auto matrix_from_gltf(std::span<double const, k_mat4_element_count> source) -> mat4
{
  mat4 matrix{};
  for (std::size_t index = 0; index < k_mat4_element_count; ++index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    matrix.cols.at(index) = static_cast<float>(source[index]);
  }
  return matrix;
}

inline auto multiply(mat4 const &lhs, mat4 const &rhs) -> mat4
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

inline auto matrix_from_translation(vec3 translation) -> mat4
{
  mat4 matrix = identity_matrix();
  matrix.cols.at((k_translation_col * k_matrix_stride) + k_axis_x) = translation.x;
  matrix.cols.at((k_translation_col * k_matrix_stride) + k_axis_y) = translation.y;
  matrix.cols.at((k_translation_col * k_matrix_stride) + k_axis_z) = translation.z;
  return matrix;
}

inline auto matrix_from_scale(vec3 scale) -> mat4
{
  mat4 matrix = identity_matrix();
  matrix.cols.at(k_matrix_diag_xx) = scale.x;
  matrix.cols.at(k_matrix_diag_yy) = scale.y;
  matrix.cols.at(k_matrix_diag_zz) = scale.z;
  return matrix;
}

inline auto matrix_from_quaternion(float coord_x, float coord_y, float coord_z, float coord_w) -> mat4
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

inline auto vec3_from_gltf(double const *source, vec3 fallback) -> vec3
{
  if (source == nullptr) { return fallback; }
  return vec3{
    .x = static_cast<float>(source[0]),// NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    .y = static_cast<float>(source[1]),// NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    .z = static_cast<float>(source[2]),// NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  };
}

inline auto matrix_from_trs(tg3_node const &node) -> mat4
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

inline auto mat_at(mat4 const &matrix, std::size_t column, std::size_t row) -> float
{ return matrix.cols.at((column * k_matrix_stride) + row); }

inline auto transform_point(mat4 const &matrix, vec3 point) -> vec3
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

inline auto transform_normal(mat4 const &matrix, vec3 normal) -> vec3
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

inline auto normalize_vec3(vec3 value) -> vec3
{
  float const length = std::sqrt((value.x * value.x) + (value.y * value.y) + (value.z * value.z));
  if (length <= 0.0F) { return value; }
  return vec3{ .x = value.x / length, .y = value.y / length, .z = value.z / length };
}

inline auto shade_vertex(vec3 normal, std::array<float, k_mesh_vertex_components> const &base_color)
  -> std::array<float, k_mesh_vertex_components>
{
  constexpr auto k_inv_sqrt3 = std::numbers::inv_sqrt3_v<float>;
  float const n_dot_l = std::max(0.0F, (normal.x * k_inv_sqrt3) + (normal.y * k_inv_sqrt3) + (normal.z * k_inv_sqrt3));
  float const intensity = k_ambient + ((k_unit_scale - k_ambient) * n_dot_l);
  return {
    base_color.at(0) * intensity,
    base_color.at(1) * intensity,
    base_color.at(2) * intensity,
  };
}

inline auto node_local_matrix(tg3_node const &node) -> mat4
{
  if (node.has_matrix != 0) { return matrix_from_gltf(std::span<double const, k_mat4_element_count>(node.matrix)); }
  return matrix_from_trs(node);
}

inline auto material_color(tg3_model const &model, std::int32_t material_index)
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

}// namespace vkexec::examples::detail

#endif// VKEXEC_EXAMPLES_LOAD_GLTF_MESH_MATH_HPP
