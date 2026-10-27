#include "load_gltf_mesh.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <span>
#include <string>
#include <tiny_gltf_v3.h>
#include <utility>
#include <vector>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>
#include <vkexec_graphics/mesh.hpp>

#include "load_gltf_mesh_math.hpp"

namespace vkexec::examples::detail {

static auto accessor_at(tg3_model const &model, std::int32_t accessor_index) -> vkexec::result<tg3_accessor const *>
{
  if (accessor_index < 0) { return fail(errc::out_of_range, "gltf accessor index out of range"); }
  if (std::cmp_greater_equal(accessor_index, model.accessors_count)) {
    return fail(errc::out_of_range, "gltf accessor index out of range");
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  return &model.accessors[static_cast<std::size_t>(accessor_index)];
}

static auto buffer_view_at(tg3_model const &model, std::int32_t buffer_view_index)
  -> vkexec::result<tg3_buffer_view const *>
{
  if (buffer_view_index < 0) { return fail(errc::out_of_range, "gltf buffer view index out of range"); }
  if (std::cmp_greater_equal(buffer_view_index, model.buffer_views_count)) {
    return fail(errc::out_of_range, "gltf buffer view index out of range");
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  return &model.buffer_views[static_cast<std::size_t>(buffer_view_index)];
}

static auto buffer_at(tg3_model const &model, std::int32_t buffer_index) -> vkexec::result<tg3_buffer const *>
{
  if (buffer_index < 0) { return fail(errc::out_of_range, "gltf buffer index out of range"); }
  if (std::cmp_greater_equal(buffer_index, model.buffers_count)) {
    return fail(errc::out_of_range, "gltf buffer index out of range");
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  return &model.buffers[static_cast<std::size_t>(buffer_index)];
}

static auto accessor_bytes(tg3_model const &model, std::int32_t accessor_index)
  -> vkexec::result<std::span<std::uint8_t const>>
{
  if (accessor_index < 0 || std::cmp_greater_equal(accessor_index, model.accessors_count)) {
    return fail(errc::out_of_range, "gltf accessor index out of range");
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  tg3_accessor const &accessor = model.accessors[static_cast<std::size_t>(accessor_index)];
  if (accessor.buffer_view < 0) { return fail(errc::parse_error, "gltf accessor missing buffer view"); }
  if (std::cmp_greater_equal(accessor.buffer_view, model.buffer_views_count)) {
    return fail(errc::out_of_range, "gltf buffer view index out of range");
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  tg3_buffer_view const &view = model.buffer_views[static_cast<std::size_t>(accessor.buffer_view)];
  if (view.buffer < 0) { return fail(errc::parse_error, "gltf buffer view missing buffer"); }
  if (std::cmp_greater_equal(view.buffer, model.buffers_count)) {
    return fail(errc::out_of_range, "gltf buffer index out of range");
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  tg3_buffer const &buffer = model.buffers[static_cast<std::size_t>(view.buffer)];
  std::size_t const offset =
    static_cast<std::size_t>(view.byte_offset) + static_cast<std::size_t>(accessor.byte_offset);
  int const component_size = tg3_component_size(accessor.component_type);
  int const component_count = tg3_num_components(accessor.type);
  if (component_size < 0 || component_count < 0) {
    return fail(errc::parse_error, "gltf accessor has invalid component layout");
  }
  std::size_t const byte_length = static_cast<std::size_t>(accessor.count) * static_cast<std::size_t>(component_size)
                                  * static_cast<std::size_t>(component_count);
  if (offset + byte_length > buffer.data.count) {
    return fail(errc::parse_error, "gltf accessor exceeds buffer bounds");
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  return std::span<std::uint8_t const>{ buffer.data.data + offset, byte_length };
}

static auto read_vec3_positions(tg3_model const &model, std::int32_t accessor_index)
  -> vkexec::result<std::vector<vec3>>
{
  auto accessor_result = accessor_at(model, accessor_index);
  if (!accessor_result) { return fail(accessor_result); }
  tg3_accessor const &accessor = *accessor_result.value();

  if (accessor.type != k_gltf_type_vec3 || accessor.component_type != k_gltf_component_float) {
    return fail(errc::parse_error, "gltf POSITION accessor must be VEC3/float");
  }
  if (accessor.buffer_view < 0) { return fail(errc::parse_error, "gltf POSITION accessor missing buffer view"); }
  auto view_result = buffer_view_at(model, accessor.buffer_view);
  if (!view_result) { return fail(view_result); }
  tg3_buffer_view const &view = *view_result.value();

  if (view.buffer < 0) { return fail(errc::parse_error, "gltf POSITION buffer view missing buffer"); }
  auto buffer_result = buffer_at(model, view.buffer);
  if (!buffer_result) { return fail(buffer_result); }
  tg3_buffer const &buffer = *buffer_result.value();

  int const stride = tg3_accessor_byte_stride(&accessor, &view);
  if (stride <= 0) { return fail(errc::parse_error, "gltf POSITION accessor has invalid byte stride"); }

  std::span<std::uint8_t const> const bytes(buffer.data.data, buffer.data.count);
  std::size_t const base_offset =
    static_cast<std::size_t>(view.byte_offset) + static_cast<std::size_t>(accessor.byte_offset);
  std::vector<vec3> positions(static_cast<std::size_t>(accessor.count));
  for (std::size_t index = 0; index < positions.size(); ++index) {
    std::array<float, k_mesh_vertex_components> values{};
    std::size_t const byte_offset = base_offset + (index * static_cast<std::size_t>(stride));
    if (byte_offset + sizeof(values) > bytes.size()) {
      return fail(errc::parse_error, "gltf POSITION accessor exceeds buffer bounds");
    }
    std::memcpy(values.data(), bytes.subspan(byte_offset, sizeof(values)).data(), sizeof(values));
    positions.at(index) = vec3{ .x = values.at(0), .y = values.at(1), .z = values.at(2) };
  }
  return positions;
}

static auto read_vec3_normals(tg3_model const &model, std::int32_t accessor_index, std::size_t expected_count)
  -> vkexec::result<std::vector<vec3>>
{
  auto accessor_result = accessor_at(model, accessor_index);
  if (!accessor_result) { return fail(accessor_result); }
  tg3_accessor const &accessor = *accessor_result.value();

  if (accessor.type != k_gltf_type_vec3 || accessor.component_type != k_gltf_component_float) {
    return fail(errc::parse_error, "gltf NORMAL accessor must be VEC3/float");
  }
  if (accessor.buffer_view < 0) { return fail(errc::parse_error, "gltf NORMAL accessor missing buffer view"); }
  auto view_result = buffer_view_at(model, accessor.buffer_view);
  if (!view_result) { return fail(view_result); }
  tg3_buffer_view const &view = *view_result.value();

  if (view.buffer < 0) { return fail(errc::parse_error, "gltf NORMAL buffer view missing buffer"); }
  auto buffer_result = buffer_at(model, view.buffer);
  if (!buffer_result) { return fail(buffer_result); }
  tg3_buffer const &buffer = *buffer_result.value();

  int const stride = tg3_accessor_byte_stride(&accessor, &view);
  if (stride <= 0) { return fail(errc::parse_error, "gltf NORMAL accessor has invalid byte stride"); }

  std::span<std::uint8_t const> const bytes(buffer.data.data, buffer.data.count);
  std::size_t const base_offset =
    static_cast<std::size_t>(view.byte_offset) + static_cast<std::size_t>(accessor.byte_offset);
  std::size_t const count = std::min(static_cast<std::size_t>(accessor.count), expected_count);
  std::vector<vec3> normals(count);
  for (std::size_t index = 0; index < count; ++index) {
    std::array<float, k_mesh_vertex_components> values{};
    std::size_t const byte_offset = base_offset + (index * static_cast<std::size_t>(stride));
    if (byte_offset + sizeof(values) > bytes.size()) {
      return fail(errc::parse_error, "gltf NORMAL accessor exceeds buffer bounds");
    }
    std::memcpy(values.data(), bytes.subspan(byte_offset, sizeof(values)).data(), sizeof(values));
    normals.at(index) = vec3{ .x = values.at(0), .y = values.at(1), .z = values.at(2) };
  }
  return normals;
}

static auto read_vec2_texcoords(tg3_model const &model, std::int32_t accessor_index, std::size_t expected_count)
  -> vkexec::result<std::vector<std::array<float, 2>>>
{
  auto accessor_result = accessor_at(model, accessor_index);
  if (!accessor_result) { return fail(accessor_result); }
  tg3_accessor const &accessor = *accessor_result.value();
  if (accessor.type != k_gltf_type_vec2 || accessor.component_type != k_gltf_component_float
      || static_cast<std::size_t>(accessor.count) != expected_count) {
    return fail(errc::parse_error, "gltf TEXCOORD_0 accessor must be VEC2/float with one value per vertex");
  }
  auto view_result = buffer_view_at(model, accessor.buffer_view);
  if (!view_result) { return fail(view_result); }
  tg3_buffer_view const &view = *view_result.value();
  auto buffer_result = buffer_at(model, view.buffer);
  if (!buffer_result) { return fail(buffer_result); }
  tg3_buffer const &buffer = *buffer_result.value();
  int const stride = tg3_accessor_byte_stride(&accessor, &view);
  if (stride <= 0) { return fail(errc::parse_error, "gltf TEXCOORD_0 accessor has invalid stride"); }
  std::span<std::uint8_t const> const bytes(buffer.data.data, buffer.data.count);
  std::size_t const base_offset =
    static_cast<std::size_t>(view.byte_offset) + static_cast<std::size_t>(accessor.byte_offset);
  std::vector<std::array<float, 2>> texcoords(expected_count);
  for (std::size_t index = 0; index < expected_count; ++index) {
    std::size_t const offset = base_offset + (index * static_cast<std::size_t>(stride));
    if (offset + sizeof(texcoords.front()) > bytes.size()) {
      return fail(errc::parse_error, "gltf TEXCOORD_0 accessor exceeds buffer bounds");
    }
    std::memcpy(
      texcoords.at(index).data(), bytes.subspan(offset, sizeof(texcoords.front())).data(), sizeof(texcoords.front()));
  }
  return texcoords;
}

static auto read_indices(tg3_model const &model, std::int32_t accessor_index)
  -> vkexec::result<std::vector<std::uint32_t>>
{
  auto accessor_result = accessor_at(model, accessor_index);
  if (!accessor_result) { return fail(accessor_result); }
  tg3_accessor const &accessor = *accessor_result.value();

  if (accessor.type != k_gltf_type_scalar
      || (accessor.component_type != k_gltf_component_unsigned_int
          && accessor.component_type != k_gltf_component_unsigned_short
          && accessor.component_type != k_gltf_component_unsigned_byte)) {
    return fail(errc::parse_error, "gltf indices accessor must be unsigned scalar");
  }
  auto bytes_result = accessor_bytes(model, accessor_index);
  if (!bytes_result) { return fail(bytes_result); }
  std::span<std::uint8_t const> const bytes = bytes_result.value();

  std::vector<std::uint32_t> indices(static_cast<std::size_t>(accessor.count));
  for (std::size_t index = 0; index < indices.size(); ++index) {
    if (accessor.component_type == k_gltf_component_unsigned_int) {
      std::uint32_t value = 0;
      std::memcpy(&value, bytes.subspan(index * sizeof(value), sizeof(value)).data(), sizeof(value));
      indices.at(index) = value;
    } else if (accessor.component_type == k_gltf_component_unsigned_short) {
      std::uint16_t value = 0;
      std::memcpy(&value, bytes.subspan(index * sizeof(value), sizeof(value)).data(), sizeof(value));
      indices.at(index) = value;
    } else {
      indices.at(index) = bytes.subspan(index, 1).front();
    }
  }
  return indices;
}

static auto find_primitive_attribute(tg3_primitive const &primitive, char const *name) -> std::int32_t
{
  for (std::uint32_t index = 0; index < primitive.attributes_count; ++index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    tg3_str_int_pair const &attribute = primitive.attributes[index];
    if (tg3_str_equals_cstr(attribute.key, name) != 0) { return attribute.value; }
  }
  return TG3_INDEX_NONE;
}

static auto append_primitive(tg3_model const &model,
  tg3_primitive const &primitive,
  mat4 const &world,
  gltf_mesh_data &mesh_data,
  bool bake_vertex_lighting) -> vkexec::status
{
  auto &vertices = mesh_data.vertices;
  auto &indices = mesh_data.indices;
  int const mode = primitive.mode < 0 ? k_gltf_mode_triangles : primitive.mode;
  if (mode != k_gltf_mode_triangles) { return {}; }

  std::int32_t const position_accessor = find_primitive_attribute(primitive, "POSITION");
  if (position_accessor < 0) { return fail(errc::parse_error, "gltf primitive missing POSITION"); }
  if (primitive.indices < 0) { return fail(errc::parse_error, "gltf primitive missing indices"); }

  auto local_positions_result = read_vec3_positions(model, position_accessor);
  if (!local_positions_result) { return fail(local_positions_result); }
  std::vector<vec3> const &local_positions = local_positions_result.value();

  auto local_indices_result = read_indices(model, primitive.indices);
  if (!local_indices_result) { return fail(local_indices_result); }
  std::vector<std::uint32_t> const &local_indices = local_indices_result.value();

  std::array<float, k_mesh_vertex_components> const base_color = material_color(model, primitive.material);

  std::int32_t const normal_accessor = find_primitive_attribute(primitive, "NORMAL");
  std::vector<vec3> local_normals;
  if (normal_accessor >= 0) {
    auto local_normals_result = read_vec3_normals(model, normal_accessor, local_positions.size());
    if (!local_normals_result) { return fail(local_normals_result); }
    local_normals = std::move(local_normals_result.value());
  }
  bool const has_normals = local_normals.size() == local_positions.size();

  std::vector<std::array<float, 2>> texcoords;
  std::int32_t const texcoord_accessor = find_primitive_attribute(primitive, "TEXCOORD_0");
  if (texcoord_accessor >= 0) {
    auto texcoords_result = read_vec2_texcoords(model, texcoord_accessor, local_positions.size());
    if (!texcoords_result) { return fail(texcoords_result); }
    texcoords = std::move(texcoords_result.value());
  }

  auto const first_index = static_cast<std::uint32_t>(indices.size());

  auto const base_vertex = static_cast<std::uint32_t>(vertices.size());
  vertices.reserve(vertices.size() + local_positions.size());
  for (std::size_t index = 0; index < local_positions.size(); ++index) {
    vec3 const world_position = transform_point(world, local_positions.at(index));
    std::array<float, k_mesh_vertex_components> color = base_color;
    std::array<float, k_mesh_vertex_components> normal{ 0.0F, 0.0F, 1.0F };
    if (has_normals) {
      vec3 const world_normal = normalize_vec3(transform_normal(world, local_normals.at(index)));
      if (bake_vertex_lighting) { color = shade_vertex(world_normal, base_color); }
      normal = { world_normal.x, world_normal.y, world_normal.z };
    }
    vertices.push_back(mesh_vertex{
      .position = { world_position.x, world_position.y, world_position.z },
      .color = color,
      .normal = normal,
      .texcoord = texcoords.empty() ? std::array<float, 2>{} : texcoords.at(index),
    });
  }

  indices.reserve(indices.size() + local_indices.size());
  std::ranges::transform(local_indices,
    std::back_inserter(indices),
    [base_vertex](std::uint32_t local_index) -> std::uint32_t { return base_vertex + local_index; });
  mesh_data.draws.push_back(gltf_mesh_data::primitive_draw{
    .first_index = first_index,
    .index_count = static_cast<std::uint32_t>(local_indices.size()),
    .material_index = primitive.material,
  });
  return {};
}

static auto append_mesh(tg3_model const &model,
  std::int32_t mesh_index,
  mat4 const &world,
  gltf_mesh_data &mesh_data,
  bool bake_vertex_lighting) -> vkexec::status
{
  if (mesh_index < 0) { return {}; }
  if (std::cmp_greater_equal(mesh_index, model.meshes_count)) { return {}; }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  tg3_mesh const &mesh = model.meshes[static_cast<std::size_t>(mesh_index)];
  for (std::uint32_t primitive_index = 0; primitive_index < mesh.primitives_count; ++primitive_index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    tg3_primitive const &primitive = mesh.primitives[primitive_index];
    if (auto status = append_primitive(model, primitive, world, mesh_data, bake_vertex_lighting); !status) {
      return fail(status);
    }
  }
  return {};
}

static auto traverse_nodes(tg3_model const &model,
  std::int32_t node_index,
  mat4 const &parent,
  gltf_mesh_data &mesh_data,
  bool bake_vertex_lighting) -> vkexec::status
{
  if (node_index < 0) { return {}; }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  if (std::cmp_greater_equal(node_index, model.nodes_count)) { return {}; }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  tg3_node const &node = model.nodes[static_cast<std::size_t>(node_index)];
  mat4 const world = multiply(parent, node_local_matrix(node));
  if (auto status = append_mesh(model, node.mesh, world, mesh_data, bake_vertex_lighting); !status) {
    return fail(status);
  }
  for (std::uint32_t child_index = 0; child_index < node.children_count; ++child_index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    if (auto status = traverse_nodes(model, node.children[child_index], world, mesh_data, bake_vertex_lighting);
      !status) {
      return fail(status);
    }
  }
  return {};
}

static auto fit_mesh_to_clip_space(std::vector<mesh_vertex> &vertices) -> void
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

static auto format_tg3_errors(tg3_error_stack const *errors) -> std::string
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

}// namespace vkexec::examples::detail

namespace vkexec::examples {

auto detail::load_gltf_mesh_factory::operator()() const -> vkexec::result<gltf_mesh_data>
{
  tinygltf3::Model model;
  tinygltf3::ErrorStack errors;
  tg3_parse_options options{};
  tg3_parse_options_init(&options);

  tg3_error_code const parse_error =
    tg3_parse_file(model.get(), errors.get(), path.c_str(), static_cast<std::uint32_t>(path.size()), &options);
  if (parse_error != TG3_OK || errors.has_error()) {
    return fail(errc::io_error, "tinygltf failed to load " + path + ": " + format_tg3_errors(errors.get()));
  }

  tg3_model const &gltf = *model.get();
  gltf_mesh_data mesh_data{};
  mat4 const identity = identity_matrix();
  if (gltf.scenes_count == 0) { return fail(errc::parse_error, "gltf file has no scenes"); }
  int const scene_index = gltf.default_scene >= 0 ? gltf.default_scene : 0;
  if (std::cmp_greater_equal(scene_index, gltf.scenes_count)) {
    return fail(errc::parse_error, "gltf default scene index out of range");
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  tg3_scene const &scene = gltf.scenes[static_cast<std::size_t>(scene_index)];
  for (std::uint32_t node_index = 0; node_index < scene.nodes_count; ++node_index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    if (auto status = traverse_nodes(gltf, scene.nodes[node_index], identity, mesh_data, fit_to_clip_space); !status) {
      return fail(status);
    }
  }

  mesh_data.base_color_textures.reserve(gltf.materials_count);
  for (std::uint32_t index = 0; index < gltf.materials_count; ++index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    tg3_material const &material = gltf.materials[index];
    std::int32_t const texture_index = material.pbr_metallic_roughness.base_color_texture.index;
    if (texture_index < 0 || std::cmp_greater_equal(texture_index, gltf.textures_count)) {
      mesh_data.base_color_textures.emplace_back();
      continue;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::int32_t const image_index = gltf.textures[static_cast<std::size_t>(texture_index)].source;
    if (image_index < 0 || std::cmp_greater_equal(image_index, gltf.images_count)) {
      return fail(errc::parse_error, "gltf material texture has no image");
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    tg3_str const uri = gltf.images[static_cast<std::size_t>(image_index)].uri;
    mesh_data.base_color_textures.emplace_back(uri.data, uri.len);
  }

  if (mesh_data.vertices.empty() || mesh_data.indices.empty()) {
    return fail(errc::empty_result, "gltf file contained no triangle geometry");
  }
  if (fit_to_clip_space) { fit_mesh_to_clip_space(mesh_data.vertices); }
  return mesh_data;
}

}// namespace vkexec::examples
