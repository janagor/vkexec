#ifndef VKEXEC_EXAMPLES_ASYNC_COMPUTE_SCENE_HPP
#define VKEXEC_EXAMPLES_ASYNC_COMPUTE_SCENE_HPP

#include "../common/free_camera.hpp"
#include "../common/ktx_texture.hpp"
#include "../common/sampler.hpp"
#include "common/shader_loader.hpp"
#include "common/smoke_frames.hpp"
#include "common/vulkan_requirements.hpp"
#include "glfw_presenter.hpp"
#include "load_gltf_mesh.hpp"
#include "sync_wait_helpers.hpp"

#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/image_view.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/result.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>
#include <vkexec_graphics/mesh.hpp>
#include <vkexec_graphics/present.hpp>
#include <vkexec_graphics/presenter.hpp>
#include <vkexec_vma/allocator.hpp>
#include <vkexec_vma/gpu_buffer.hpp>
#include <vkexec_vma/image.hpp>
#include <vkexec_vma/offscreen_target.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vkexec::examples::async_compute {

constexpr std::uint32_t k_window_width = 800;
constexpr std::uint32_t k_window_height = 600;
constexpr std::uint32_t k_shadow_extent = 4096;
constexpr std::uint32_t k_hdr_scale = 2;
constexpr std::uint32_t k_bloom_levels = 4;
constexpr std::uint32_t k_bloom_image_count = (k_bloom_levels * 2) - 1;
constexpr std::uint32_t k_local_size = 8;
constexpr std::uint32_t k_fullscreen_vertices = 3;
constexpr std::size_t k_frames_in_flight = 2;
constexpr std::size_t k_matrix_stride = 4;
constexpr float k_shadow_radius_scale = 1.3F;
constexpr float k_camera_distance_scale = 1.5F;
constexpr float k_camera_height_scale = 0.25F;
constexpr float k_camera_speed_scale = 0.35F;
constexpr float k_camera_far_scale = 8.0F;
constexpr float k_depth_scale = 0.5F;
constexpr std::array<float, 3> k_light_direction{ -0.4F, -1.0F, -0.3F };
constexpr std::array<float, 4> k_light_color{ 3.5F, 3.0F, 2.5F, 0.0F };
constexpr VkFormat k_hdr_format = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat k_depth_format = VK_FORMAT_D32_SFLOAT;

enum class queue_mode : std::uint8_t { single, async };

struct scene_bounds
{
  std::array<float, 3> center{};
  float radius{ 1.0F };
};

struct light_data
{
  std::array<float, k_matrix_stride * k_matrix_stride> matrix{};
  std::array<float, 4> direction{};
  std::array<float, 4> color{};
};

struct bloom_params
{
  std::array<std::uint32_t, 2> resolution{};
  std::array<float, 2> inv_resolution{};
  std::array<float, 2> inv_input_resolution{};
};

struct bloom_image
{
  vkexec::vma::image image;
  vkexec::owned::image_view view;
  VkExtent2D extent{};
};

struct compute_pipelines
{
  vkexec::owned::compute_pipeline threshold;
  vkexec::owned::compute_pipeline down;
  vkexec::owned::compute_pipeline up;
};

struct bloom_pass
{
  vkexec::owned::compute_pipeline const *pipeline{ nullptr };
  VkDescriptorSet set{ VK_NULL_HANDLE };
  vkexec::resource_table resources;
  std::size_t output_index{ 0 };
  VkExtent2D input_extent{};
};

struct frame_resources
{
  vkexec::vma::offscreen_target shadow;
  vkexec::vma::offscreen_target hdr;
  std::vector<bloom_image> bloom;
  vkexec::vma::gpu_buffer camera_buffer;
  vkexec::vma::gpu_buffer light_buffer;
  vkexec::owned::graphics_pipeline shadow_pipeline;
  std::vector<vkexec::owned::graphics_pipeline> forward;
  vkexec::owned::graphics_pipeline composite;
  std::vector<bloom_pass> passes;
  bool initialized{ false };
};

inline auto check(vkexec::status result) -> void
{
  if (!result) { vkexec::examples::abort_with_error(result.error()); }
}

[[nodiscard]] inline auto find_bounds(std::span<vkexec::mesh_vertex const> vertices) -> scene_bounds
{
  if (vertices.empty()) { vkexec::examples::fail_check("Bonza has no vertices"); }
  std::array<float, 3> minimum{};
  std::array<float, 3> maximum{};
  minimum.fill(std::numeric_limits<float>::max());
  maximum.fill(std::numeric_limits<float>::lowest());
  for (vkexec::mesh_vertex const &vertex : vertices) {
    for (std::size_t axis = 0; axis < minimum.size(); ++axis) {
      minimum.at(axis) = std::min(minimum.at(axis), vertex.position.at(axis));
      maximum.at(axis) = std::max(maximum.at(axis), vertex.position.at(axis));
    }
  }
  scene_bounds result{};
  for (std::size_t axis = 0; axis < result.center.size(); ++axis) {
    result.center.at(axis) = (minimum.at(axis) + maximum.at(axis)) * k_depth_scale;
    result.radius = std::max(result.radius, (maximum.at(axis) - minimum.at(axis)) * k_depth_scale);
  }
  return result;
}

[[nodiscard]] inline auto dot(std::array<float, 3> const &left, std::array<float, 3> const &right) -> float
{
  float value = 0.0F;
  for (std::size_t axis = 0; axis < left.size(); ++axis) { value += left.at(axis) * right.at(axis); }
  return value;
}

[[nodiscard]] inline auto normalize(std::array<float, 3> value) -> std::array<float, 3>
{
  float const inverse_length = 1.0F / std::sqrt(dot(value, value));
  std::ranges::transform(
    value, value.begin(), [inverse_length](float component) -> float { return component * inverse_length; });
  return value;
}

[[nodiscard]] inline auto make_light(scene_bounds const &bounds) -> light_data
{
  auto const direction = normalize(k_light_direction);
  auto const right = normalize(std::array<float, 3>{ -direction.at(2), 0.0F, direction.at(0) });
  auto const light_up = normalize(std::array<float, 3>{
    (right.at(1) * direction.at(2)) - (right.at(2) * direction.at(1)),
    (right.at(2) * direction.at(0)) - (right.at(0) * direction.at(2)),
    (right.at(0) * direction.at(1)) - (right.at(1) * direction.at(0)),
  });
  float const radius = bounds.radius * k_shadow_radius_scale;
  light_data light{};
  for (std::size_t axis = 0; axis < bounds.center.size(); ++axis) {
    light.matrix.at((axis * k_matrix_stride) + 0) = right.at(axis) / radius;
    light.matrix.at((axis * k_matrix_stride) + 1) = -light_up.at(axis) / radius;
    light.matrix.at((axis * k_matrix_stride) + 2) = direction.at(axis) * k_depth_scale / radius;
  }
  light.matrix.at((3 * k_matrix_stride) + 0) = -dot(bounds.center, right) / radius;
  light.matrix.at((3 * k_matrix_stride) + 1) = dot(bounds.center, light_up) / radius;
  light.matrix.at((3 * k_matrix_stride) + 2) = k_depth_scale - (dot(bounds.center, direction) * k_depth_scale / radius);
  light.matrix.at((3 * k_matrix_stride) + 3) = 1.0F;
  light.direction = { direction.at(0), direction.at(1), direction.at(2), 0.0F };
  light.color = k_light_color;
  return light;
}

[[nodiscard]] inline auto scaled_extent(VkExtent2D extent) -> VkExtent2D
{ return VkExtent2D{ .width = extent.width * k_hdr_scale, .height = extent.height * k_hdr_scale }; }

[[nodiscard]] inline auto half_extent(VkExtent2D extent) -> VkExtent2D
{ return VkExtent2D{ .width = std::max(1U, extent.width / 2), .height = std::max(1U, extent.height / 2) }; }

[[nodiscard]] inline auto make_bloom_image(vkexec::context &ctx, vkexec::vma::allocator &allocator, VkExtent2D extent)
  -> bloom_image
{
  auto image = vkexec::examples::sync_wait_value(vkexec::allocate_image(allocator,
    vkexec::image_create_info{
      .extent = { .width = extent.width, .height = extent.height, .depth = 1 },
      .format = k_hdr_format,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
      .queue_families = {},
    }));
  auto view = vkexec::examples::sync_wait_value(vkexec::vma::factory::make_image_view(ctx, image));
  return bloom_image{ .image = std::move(image), .view = std::move(view), .extent = extent };
}

[[nodiscard]] inline auto make_compute_pipelines(vkexec::context &ctx, std::filesystem::path const &shader_dir)
  -> compute_pipelines
{
  using enum vkexec::buffer_access;
  vkexec::layout_desc const layout{
    .binding_kinds = { vkexec::resource_kind::sampled_image,
      vkexec::resource_kind::storage_image,
      vkexec::resource_kind::sampler },
    .binding_slots = {},
    .bindings = { readonly, writeonly, readonly },
    .push_constant_size = sizeof(bloom_params),
    .specialization = {},
    .local_size = { k_local_size, k_local_size, 1 },
  };
  return compute_pipelines{
    .threshold = vkexec::examples::sync_wait_value(vkexec::factory::make_compute_pipeline(
      ctx, vkexec::examples::load_spirv(shader_dir / "threshold.comp.spv"), layout)),
    .down = vkexec::examples::sync_wait_value(vkexec::factory::make_compute_pipeline(
      ctx, vkexec::examples::load_spirv(shader_dir / "blur_down.comp.spv"), layout)),
    .up = vkexec::examples::sync_wait_value(vkexec::factory::make_compute_pipeline(
      ctx, vkexec::examples::load_spirv(shader_dir / "blur_up.comp.spv"), layout)),
  };
}

[[nodiscard]] constexpr auto color_range() -> VkImageSubresourceRange
{
  return {
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1
  };
}

[[nodiscard]] inline auto make_frame(vkexec::examples::glfw_presenter &win,
  vkexec::vma::allocator &allocator,
  std::filesystem::path const &shader_dir,
  std::vector<vkexec::examples::ktx_texture> const &textures,
  vkexec::examples::sampler_owner const &material_sampler,
  vkexec::examples::sampler_owner const &bloom_sampler,
  compute_pipelines const &compute,
  light_data const &light) -> frame_resources
{
  auto &ctx = win.ctx();
  auto shadow_result = vkexec::vma::offscreen_target::create(ctx,
    allocator,
    vkexec::vma::offscreen_target_config{
      .extent = { .width = k_shadow_extent, .height = k_shadow_extent },
      .color_formats = {},
      .depth_format = k_depth_format,
      .sample_depth = true,
    });
  if (!shadow_result) { vkexec::examples::abort_with_error(shadow_result.error()); }
  auto shadow = std::move(*shadow_result);
  auto hdr_result = vkexec::vma::offscreen_target::create(ctx,
    allocator,
    vkexec::vma::offscreen_target_config{
      .extent = scaled_extent(win.target().extent()),
      .color_formats = { k_hdr_format },
      .depth_format = k_depth_format,
    });
  if (!hdr_result) { vkexec::examples::abort_with_error(hdr_result.error()); }
  auto hdr = std::move(*hdr_result);

  std::vector<bloom_image> bloom;
  bloom.reserve(k_bloom_image_count);
  std::array<VkExtent2D, k_bloom_levels> levels{};
  VkExtent2D level_extent = hdr.extent();
  for (VkExtent2D &level : levels) {
    level_extent = half_extent(level_extent);
    level = level_extent;
    bloom.push_back(make_bloom_image(ctx, allocator, level));
  }
  for (std::size_t level = k_bloom_levels - 1; level > 0; --level) {
    bloom.push_back(make_bloom_image(ctx, allocator, levels.at(level - 1)));
  }

  auto camera_buffer = vkexec::examples::sync_wait_value(vkexec::vma::factory::make_gpu_buffer(allocator,
    vkexec::vma::gpu_buffer_create_info{
      .size = sizeof(vkexec::examples::camera_data),
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .memory = vkexec::vma::buffer_memory::host_visible,
    }));
  auto light_buffer = vkexec::examples::sync_wait_value(vkexec::vma::factory::make_gpu_buffer(allocator,
    vkexec::vma::gpu_buffer_create_info{
      .size = sizeof(light_data),
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .memory = vkexec::vma::buffer_memory::host_visible,
    }));
  std::memcpy(light_buffer.mapped().data(), &light, sizeof(light));
  check(light_buffer.flush());

  vkexec::graphics_pipeline_config shadow_config{};
  shadow_config.depth_test = true;
  shadow_config.use_mesh_vertices = true;
  shadow_config.mesh_position_only = true;
  shadow_config.color_attachment_count = 0;
  auto shadow_pipeline = vkexec::examples::sync_wait_value(vkexec::factory::make_graphics_pipeline(ctx,
    shadow.render_pass(),
    shadow_config,
    vkexec::examples::load_spirv(shader_dir / "shadow.vert.spv"),
    vkexec::examples::load_spirv(shader_dir / "shadow.frag.spv"),
    vkexec::bindings(vkexec::resource_binding{
      .slot = 0, .resource = vkexec::buffer_resource(light_buffer.handle(), sizeof(light_data)) })));

  vkexec::graphics_pipeline_config forward_config{};
  forward_config.depth_test = true;
  forward_config.use_mesh_vertices = true;
  auto const forward_vert = vkexec::examples::load_spirv(shader_dir / "forward.vert.spv");
  auto const forward_frag = vkexec::examples::load_spirv(shader_dir / "forward.frag.spv");
  std::vector<vkexec::owned::graphics_pipeline> forward;
  forward.reserve(textures.size());
  for (vkexec::examples::ktx_texture const &texture : textures) {
    forward.push_back(vkexec::examples::sync_wait_value(vkexec::factory::make_graphics_pipeline(ctx,
      hdr.render_pass(),
      forward_config,
      forward_vert,
      forward_frag,
      vkexec::bindings(
        vkexec::resource_binding{ .slot = 0, .resource = vkexec::sampled_image_resource(texture.view.handle()) },
        vkexec::resource_binding{ .slot = 1, .resource = vkexec::sampler_resource(material_sampler.handle) },
        vkexec::resource_binding{ .slot = 2,
          .resource = vkexec::buffer_resource(camera_buffer.handle(), sizeof(vkexec::examples::camera_data)) },
        vkexec::resource_binding{ .slot = 3, .resource = vkexec::sampled_image_resource(shadow.depth_view()) },
        vkexec::resource_binding{
          .slot = 4, .resource = vkexec::buffer_resource(light_buffer.handle(), sizeof(light_data)) }))));
  }
  auto composite = vkexec::examples::sync_wait_value(vkexec::factory::make_graphics_pipeline(ctx,
    win.render_pass(),
    vkexec::graphics_pipeline_config{},
    vkexec::examples::load_spirv(shader_dir / "composite.vert.spv"),
    vkexec::examples::load_spirv(shader_dir / "composite.frag.spv"),
    vkexec::bindings(
      vkexec::resource_binding{ .slot = 0, .resource = vkexec::sampled_image_resource(hdr.color_view(0)) },
      vkexec::resource_binding{ .slot = 1, .resource = vkexec::sampled_image_resource(bloom.back().view.handle()) },
      vkexec::resource_binding{ .slot = 2, .resource = vkexec::sampler_resource(bloom_sampler.handle) })));

  frame_resources frame{
    .shadow = std::move(shadow),
    .hdr = std::move(hdr),
    .bloom = std::move(bloom),
    .camera_buffer = std::move(camera_buffer),
    .light_buffer = std::move(light_buffer),
    .shadow_pipeline = std::move(shadow_pipeline),
    .forward = std::move(forward),
    .composite = std::move(composite),
    .passes = {},
  };

  auto add_pass = [&](vkexec::owned::compute_pipeline const &pipeline,
                    VkImage input_image,
                    VkImageView input_view,
                    VkExtent2D input_extent,
                    std::size_t output_index) -> void {
    auto set_result = pipeline.allocate_set();
    if (!set_result) { vkexec::examples::abort_with_error(set_result.error()); }
    VkDescriptorSet set = *set_result;
    auto resources =
      vkexec::bindings(vkexec::resource_binding{ .slot = 0,
                         .resource = vkexec::sampled_image_resource(input_image, input_view, color_range()) },
        vkexec::resource_binding{ .slot = 1,
          .resource = vkexec::storage_image_resource(
            frame.bloom.at(output_index).image.handle(), frame.bloom.at(output_index).view.handle(), color_range()) },
        vkexec::resource_binding{ .slot = 2, .resource = vkexec::sampler_resource(bloom_sampler.handle) });
    check(pipeline.update_set(set, resources));
    frame.passes.push_back(bloom_pass{ .pipeline = &pipeline,
      .set = set,
      .resources = std::move(resources),
      .output_index = output_index,
      .input_extent = input_extent });
  };
  add_pass(compute.threshold, frame.hdr.color_image(0), frame.hdr.color_view(0), frame.hdr.extent(), 0);
  for (std::size_t output = 1; output < k_bloom_levels; ++output) {
    add_pass(compute.down,
      frame.bloom.at(output - 1).image.handle(),
      frame.bloom.at(output - 1).view.handle(),
      frame.bloom.at(output - 1).extent,
      output);
  }
  for (std::size_t output = k_bloom_levels; output < frame.bloom.size(); ++output) {
    add_pass(compute.up,
      frame.bloom.at(output - 1).image.handle(),
      frame.bloom.at(output - 1).view.handle(),
      frame.bloom.at(output - 1).extent,
      output);
  }
  return frame;
}

inline auto destroy_frame(vkexec::context &ctx, frame_resources &frame) -> void
{
  for (bloom_pass const &pass : frame.passes) { vkexec::free_compute_set(ctx, pass.pipeline->resources(), pass.set); }
}

[[nodiscard]] constexpr auto depth_range() -> VkImageSubresourceRange
{
  return {
    .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1
  };
}

inline auto begin_offscreen_pass(VkCommandBuffer cmd, vkexec::vma::offscreen_target const &target) -> void
{
  auto const clears = target.clear_values();
  VkRenderPassBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  begin.renderPass = target.render_pass();
  begin.framebuffer = target.framebuffer();
  begin.renderArea = VkRect2D{ .offset = VkOffset2D{ .x = 0, .y = 0 }, .extent = target.extent() };
  begin.clearValueCount = static_cast<std::uint32_t>(clears.size());
  begin.pClearValues = clears.data();
  vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
}

}// namespace vkexec::examples::async_compute

#endif// VKEXEC_EXAMPLES_ASYNC_COMPUTE_SCENE_HPP
