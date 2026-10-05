#include "../common/free_camera.hpp"
#include "../common/ktx_texture.hpp"
#include "../common/sampler.hpp"
#include "common/shader_loader.hpp"
#include "common/vulkan_requirements.hpp"
#include "glfw_presenter.hpp"
#include "load_gltf_mesh.hpp"
#include "sync_wait_helpers.hpp"

#include <vkexec/barrier.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/image_view.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/result.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>
#include <vkexec_graphics/mesh.hpp>
#include <vkexec_graphics/presenter.hpp>
#include <vkexec_vma/allocator.hpp>
#include <vkexec_vma/gpu_buffer.hpp>
#include <vkexec_vma/image.hpp>
#include <vkexec_vma/offscreen_target.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

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
  VkCommandBuffer early_cmd{ VK_NULL_HANDLE };
  VkCommandBuffer compute_cmd{ VK_NULL_HANDLE };
  VkSemaphore graphics_done{ VK_NULL_HANDLE };
  VkSemaphore compute_done{ VK_NULL_HANDLE };
  bool initialized{ false };
};

auto check(vkexec::status result) -> void
{
  if (!result) { vkexec::examples::abort_with_error(result.error()); }
}

auto check_vk(VkResult result, char const *message) -> void
{
  if (result != VK_SUCCESS) { vkexec::examples::fail_check(message); }
}

[[nodiscard]] auto find_bounds(std::span<vkexec::mesh_vertex const> vertices) -> scene_bounds
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

[[nodiscard]] auto dot(std::array<float, 3> const &left, std::array<float, 3> const &right) -> float
{
  float value = 0.0F;
  for (std::size_t axis = 0; axis < left.size(); ++axis) { value += left.at(axis) * right.at(axis); }
  return value;
}

[[nodiscard]] auto normalize(std::array<float, 3> value) -> std::array<float, 3>
{
  float const inverse_length = 1.0F / std::sqrt(dot(value, value));
  for (float &component : value) { component *= inverse_length; }
  return value;
}

[[nodiscard]] auto make_light(scene_bounds const &bounds) -> light_data
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

[[nodiscard]] auto scaled_extent(VkExtent2D extent) -> VkExtent2D
{ return VkExtent2D{ .width = extent.width * k_hdr_scale, .height = extent.height * k_hdr_scale }; }

[[nodiscard]] auto half_extent(VkExtent2D extent) -> VkExtent2D
{ return VkExtent2D{ .width = std::max(1U, extent.width / 2), .height = std::max(1U, extent.height / 2) }; }

[[nodiscard]] auto make_bloom_image(vkexec::context &ctx, vkexec::vma::allocator &allocator, VkExtent2D extent)
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

[[nodiscard]] auto make_compute_pipelines(vkexec::context &ctx, std::filesystem::path const &shader_dir)
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

[[nodiscard]] auto make_semaphore(VkDevice device) -> VkSemaphore
{
  VkSemaphoreCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkSemaphore semaphore{ VK_NULL_HANDLE };
  check_vk(vkCreateSemaphore(device, &info, nullptr, &semaphore), "vkCreateSemaphore failed");
  return semaphore;
}

[[nodiscard]] auto make_frame(vkexec::examples::glfw_presenter &win,
  vkexec::vma::allocator &allocator,
  std::filesystem::path const &shader_dir,
  std::vector<vkexec::examples::ktx_texture> const &textures,
  vkexec::examples::sampler_owner const &material_sampler,
  vkexec::examples::sampler_owner const &bloom_sampler,
  compute_pipelines const &compute,
  light_data const &light,
  vkexec::queue_ref compute_queue) -> frame_resources
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

  auto early_result = ctx.allocate_command_buffer(ctx.graphics_queue_ref());
  if (!early_result) { vkexec::examples::abort_with_error(early_result.error()); }
  auto compute_result = ctx.allocate_command_buffer(compute_queue);
  if (!compute_result) { vkexec::examples::abort_with_error(compute_result.error()); }
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
    .early_cmd = *early_result,
    .compute_cmd = *compute_result,
    .graphics_done = make_semaphore(ctx.device()),
    .compute_done = make_semaphore(ctx.device()),
  };

  auto add_pass = [&](vkexec::owned::compute_pipeline const &pipeline,
                    VkImageView input_view,
                    VkExtent2D input_extent,
                    std::size_t output_index) -> void {
    auto set_result = pipeline.allocate_set();
    if (!set_result) { vkexec::examples::abort_with_error(set_result.error()); }
    VkDescriptorSet set = *set_result;
    check(pipeline.update_set(set,
      vkexec::bindings(vkexec::resource_binding{ .slot = 0, .resource = vkexec::sampled_image_resource(input_view) },
        vkexec::resource_binding{
          .slot = 1, .resource = vkexec::storage_image_resource(frame.bloom.at(output_index).view.handle()) },
        vkexec::resource_binding{ .slot = 2, .resource = vkexec::sampler_resource(bloom_sampler.handle) })));
    frame.passes.push_back(
      bloom_pass{ .pipeline = &pipeline, .set = set, .output_index = output_index, .input_extent = input_extent });
  };
  add_pass(compute.threshold, frame.hdr.color_view(0), frame.hdr.extent(), 0);
  for (std::size_t output = 1; output < k_bloom_levels; ++output) {
    add_pass(compute.down, frame.bloom.at(output - 1).view.handle(), frame.bloom.at(output - 1).extent, output);
  }
  for (std::size_t output = k_bloom_levels; output < frame.bloom.size(); ++output) {
    add_pass(compute.up, frame.bloom.at(output - 1).view.handle(), frame.bloom.at(output - 1).extent, output);
  }
  return frame;
}

auto destroy_frame(vkexec::context &ctx, frame_resources &frame) -> void
{
  for (bloom_pass const &pass : frame.passes) { vkexec::free_compute_set(ctx, pass.pipeline->resources(), pass.set); }
  ctx.free_command_buffer(frame.early_cmd);
  ctx.free_command_buffer(frame.compute_cmd);
  vkDestroySemaphore(ctx.device(), frame.graphics_done, nullptr);
  vkDestroySemaphore(ctx.device(), frame.compute_done, nullptr);
  frame.early_cmd = VK_NULL_HANDLE;
  frame.compute_cmd = VK_NULL_HANDLE;
  frame.graphics_done = VK_NULL_HANDLE;
  frame.compute_done = VK_NULL_HANDLE;
}

auto begin_recording(VkCommandBuffer cmd) -> void
{
  check_vk(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer failed");
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check_vk(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer failed");
}

auto transition(vkexec::context &ctx,
  VkCommandBuffer cmd,
  VkImage image,
  VkImageAspectFlags aspect,
  VkImageLayout old_layout,
  VkImageLayout new_layout,
  VkPipelineStageFlags2 source_stage,
  VkPipelineStageFlags2 destination_stage,
  VkAccessFlags2 source_access,
  VkAccessFlags2 destination_access,
  std::uint32_t source_family = VK_QUEUE_FAMILY_IGNORED,
  std::uint32_t destination_family = VK_QUEUE_FAMILY_IGNORED) -> void
{
  check(vkexec::image_barrier(ctx,
    cmd,
    vkexec::image_barrier_params{
      .image = image,
      .range = { .aspectMask = aspect, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 },
      .old_layout = old_layout,
      .new_layout = new_layout,
      .src_stage = source_stage,
      .dst_stage = destination_stage,
      .src_access = source_access,
      .dst_access = destination_access,
      .src_queue_family = source_family,
      .dst_queue_family = destination_family,
    }));
}

auto begin_offscreen_pass(VkCommandBuffer cmd, vkexec::vma::offscreen_target const &target) -> void
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

auto record_early(vkexec::context &ctx,
  frame_resources const &frame,
  vkexec::examples::gltf_mesh_data const &scene,
  VkBuffer vertices,
  VkBuffer indices,
  std::vector<vkexec::examples::ktx_texture> const &textures,
  bool upload_textures,
  vkexec::queue_ref compute_queue) -> void
{
  VkCommandBuffer cmd = frame.early_cmd;
  begin_recording(cmd);
  if (upload_textures) {
    for (vkexec::examples::ktx_texture const &texture : textures) {
      vkexec::examples::record_ktx_texture_upload(ctx, cmd, texture);
    }
  }
  transition(ctx,
    cmd,
    frame.shadow.depth_image(),
    VK_IMAGE_ASPECT_DEPTH_BIT,
    frame.initialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    frame.initialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
    frame.initialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
  begin_offscreen_pass(cmd, frame.shadow);
  for (auto const &draw : scene.draws) {
    frame.shadow_pipeline.record_draw(cmd,
      frame.shadow.extent(),
      vkexec::mesh_draw{ .vertex_buffer = vertices,
        .index_buffer = indices,
        .index_count = draw.index_count,
        .first_index = draw.first_index });
  }
  vkCmdEndRenderPass(cmd);
  transition(ctx,
    cmd,
    frame.shadow.depth_image(),
    VK_IMAGE_ASPECT_DEPTH_BIT,
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  transition(ctx,
    cmd,
    frame.hdr.color_image(0),
    VK_IMAGE_ASPECT_COLOR_BIT,
    frame.initialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    frame.initialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
    frame.initialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  transition(ctx,
    cmd,
    frame.hdr.depth_image(),
    VK_IMAGE_ASPECT_DEPTH_BIT,
    frame.initialized ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    VK_PIPELINE_STAGE_2_NONE,
    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
    VK_ACCESS_2_NONE,
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
  begin_offscreen_pass(cmd, frame.hdr);
  for (auto const &draw : scene.draws) {
    if (draw.material_index < 0 || std::cmp_greater_equal(draw.material_index, frame.forward.size())) {
      vkexec::examples::fail_check("Bonza primitive has no base-color material");
    }
    frame.forward.at(static_cast<std::size_t>(draw.material_index))
      .record_draw(cmd,
        frame.hdr.extent(),
        vkexec::mesh_draw{ .vertex_buffer = vertices,
          .index_buffer = indices,
          .index_count = draw.index_count,
          .first_index = draw.first_index });
  }
  vkCmdEndRenderPass(cmd);
  bool const transfer = ctx.graphics_queue_family() != compute_queue.family;
  transition(ctx,
    cmd,
    frame.hdr.color_image(0),
    VK_IMAGE_ASPECT_COLOR_BIT,
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
    transfer ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
    transfer ? VK_ACCESS_2_NONE : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
    transfer ? ctx.graphics_queue_family() : VK_QUEUE_FAMILY_IGNORED,
    transfer ? compute_queue.family : VK_QUEUE_FAMILY_IGNORED);
  if (transfer && frame.initialized) {
    transition(ctx,
      cmd,
      frame.bloom.back().image.handle(),
      VK_IMAGE_ASPECT_COLOR_BIT,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      VK_PIPELINE_STAGE_2_NONE,
      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
      VK_ACCESS_2_NONE,
      ctx.graphics_queue_family(),
      compute_queue.family);
  }
  check_vk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer failed for early graphics");
}

auto record_compute(vkexec::context &ctx, frame_resources const &frame, vkexec::queue_ref compute_queue) -> void
{
  VkCommandBuffer cmd = frame.compute_cmd;
  begin_recording(cmd);
  bool const transfer = ctx.graphics_queue_family() != compute_queue.family;
  if (transfer) {
    transition(ctx,
      cmd,
      frame.hdr.color_image(0),
      VK_IMAGE_ASPECT_COLOR_BIT,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_PIPELINE_STAGE_2_NONE,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_NONE,
      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
      ctx.graphics_queue_family(),
      compute_queue.family);
    if (frame.initialized) {
      transition(ctx,
        cmd,
        frame.bloom.back().image.handle(),
        VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_NONE,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        ctx.graphics_queue_family(),
        compute_queue.family);
    }
  }
  for (bloom_pass const &pass : frame.passes) {
    bloom_image const &output = frame.bloom.at(pass.output_index);
    transition(ctx,
      cmd,
      output.image.handle(),
      VK_IMAGE_ASPECT_COLOR_BIT,
      frame.initialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_GENERAL,
      VK_PIPELINE_STAGE_2_NONE,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_NONE,
      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    bloom_params const params{
      .resolution = { output.extent.width, output.extent.height },
      .inv_resolution = { 1.0F / static_cast<float>(output.extent.width),
        1.0F / static_cast<float>(output.extent.height) },
      .inv_input_resolution = { 1.0F / static_cast<float>(pass.input_extent.width),
        1.0F / static_cast<float>(pass.input_extent.height) },
    };
    vkexec::record_pass(cmd,
      pass.pipeline->bind(pass.set),
      &params,
      static_cast<std::uint32_t>(sizeof(params)),
      vkexec::dispatch{ .x = (output.extent.width + k_local_size - 1) / k_local_size,
        .y = (output.extent.height + k_local_size - 1) / k_local_size,
        .z = 1 });
    transition(ctx,
      cmd,
      output.image.handle(),
      VK_IMAGE_ASPECT_COLOR_BIT,
      VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  }
  if (transfer) {
    transition(ctx,
      cmd,
      frame.hdr.color_image(0),
      VK_IMAGE_ASPECT_COLOR_BIT,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_2_NONE,
      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
      VK_ACCESS_2_NONE,
      compute_queue.family,
      ctx.graphics_queue_family());
    transition(ctx,
      cmd,
      frame.bloom.back().image.handle(),
      VK_IMAGE_ASPECT_COLOR_BIT,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_2_NONE,
      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      VK_ACCESS_2_NONE,
      compute_queue.family,
      ctx.graphics_queue_family());
  }
  check_vk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer failed for compute");
}

auto record_final(vkexec::context &ctx,
  frame_resources const &frame,
  vkexec::frame const &present_frame,
  vkexec::examples::glfw_presenter &win,
  vkexec::queue_ref compute_queue) -> void
{
  VkCommandBuffer cmd = present_frame.command_buffer;
  if (ctx.graphics_queue_family() != compute_queue.family) {
    std::array const images{ frame.hdr.color_image(0), frame.bloom.back().image.handle() };
    for (VkImage image : images) {
      transition(ctx,
        cmd,
        image,
        VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_NONE,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        compute_queue.family,
        ctx.graphics_queue_family());
    }
  }
  frame.composite.record_pass(
    cmd, win.render_pass(), present_frame.framebuffer, present_frame.extent, k_fullscreen_vertices);
}

auto submit_frame(vkexec::context &ctx,
  frame_resources const &frame,
  vkexec::frame const &present_frame,
  vkexec::examples::glfw_presenter &win,
  vkexec::queue_ref compute_queue) -> void
{
  std::array<VkCommandBuffer, 1> const early_commands{ frame.early_cmd };
  std::array<vkexec::semaphore_submit, 1> const early_signals{
    vkexec::semaphore_submit{ .semaphore = frame.graphics_done },
  };
  check(ctx.submit(vkexec::queue_submit{
    .command_buffers = early_commands, .signals = early_signals, .queue = ctx.graphics_queue() }));
  std::array<VkCommandBuffer, 1> const compute_commands{ frame.compute_cmd };
  std::array<vkexec::semaphore_submit, 1> const compute_waits{
    vkexec::semaphore_submit{ .semaphore = frame.graphics_done, .stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT },
  };
  std::array<vkexec::semaphore_submit, 1> const compute_signals{
    vkexec::semaphore_submit{ .semaphore = frame.compute_done },
  };
  check(ctx.submit(vkexec::queue_submit{ .command_buffers = compute_commands,
    .waits = compute_waits,
    .signals = compute_signals,
    .queue = compute_queue.queue }));
  std::array<vkexec::semaphore_submit, 1> const final_waits{
    vkexec::semaphore_submit{ .semaphore = frame.compute_done, .stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT },
  };
  auto ended = win.end_frame(present_frame, vkexec::frame_submit_options{ .waits = final_waits });
  if (!ended) { vkexec::examples::abort_with_error(ended.error()); }
}

[[nodiscard]] auto parse_mode(int argc, char const *const *argv) -> queue_mode
{
  queue_mode mode = queue_mode::async;
  for (int index = 1; index < argc; ++index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::string_view const argument(argv[index]);
    if (argument == "--queues=single") {
      mode = queue_mode::single;
    } else if (argument == "--queues=async") {
      mode = queue_mode::async;
    } else {
      vkexec::examples::fail_check("use --queues=single or --queues=async");
    }
  }
  return mode;
}

auto check_image_formats(vkexec::context &ctx) -> void
{
  VkFormatProperties texture_properties{};
  vkGetPhysicalDeviceFormatProperties(ctx.physical_device(), VK_FORMAT_ASTC_5x5_SRGB_BLOCK, &texture_properties);
  constexpr VkFormatFeatureFlags k_texture_features =
    static_cast<VkFormatFeatureFlags>(VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
    | static_cast<VkFormatFeatureFlags>(VK_FORMAT_FEATURE_TRANSFER_DST_BIT);
  if ((texture_properties.optimalTilingFeatures & k_texture_features) != k_texture_features) {
    vkexec::examples::fail_check("Bonza requires sampled ASTC 5x5 sRGB textures");
  }
  VkFormatProperties hdr_properties{};
  vkGetPhysicalDeviceFormatProperties(ctx.physical_device(), k_hdr_format, &hdr_properties);
  constexpr VkFormatFeatureFlags k_hdr_features =
    static_cast<VkFormatFeatureFlags>(VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)
    | static_cast<VkFormatFeatureFlags>(VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
    | static_cast<VkFormatFeatureFlags>(VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);
  if ((hdr_properties.optimalTilingFeatures & k_hdr_features) != k_hdr_features) {
    vkexec::examples::fail_check("RGBA16F color, sampled, and storage images are required");
  }
  VkFormatProperties depth_properties{};
  vkGetPhysicalDeviceFormatProperties(ctx.physical_device(), k_depth_format, &depth_properties);
  constexpr VkFormatFeatureFlags k_depth_features =
    static_cast<VkFormatFeatureFlags>(VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
    | static_cast<VkFormatFeatureFlags>(VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
    | static_cast<VkFormatFeatureFlags>(VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
  if ((depth_properties.optimalTilingFeatures & k_depth_features) != k_depth_features) {
    vkexec::examples::fail_check("D32 depth attachments need linear sampled-image support");
  }
}

[[nodiscard]] auto choose_compute_queue(vkexec::context &ctx, queue_mode mode) -> vkexec::queue_ref
{
  vkexec::queue_ref const dedicated = ctx.compute_queue_ref();
  if (mode == queue_mode::async && dedicated.queue != ctx.graphics_queue()
      && dedicated.family != ctx.graphics_queue_family()) {
    return dedicated;
  }
  return ctx.graphics_queue_ref();
}

auto print_queue_mode(vkexec::context &ctx, queue_mode requested_mode, vkexec::queue_ref compute_queue) -> void
{
  char const *mode_name = "shared queue fallback";
  if (requested_mode == queue_mode::single) {
    mode_name = "single queue";
  } else if (compute_queue.family != ctx.graphics_queue_family()) {
    mode_name = "dedicated async compute";
  }
  std::cout << "graphics family: " << ctx.graphics_queue_family() << "\ncompute family: " << compute_queue.family
            << "\nmode: " << mode_name
            << "\nWASD move, Q/E descend/ascend, right mouse drag to look; close the window to exit\n";
}

// NOLINTNEXTLINE(bugprone-exception-escape)
auto run(int argc, char const *const *argv) -> int
{
  queue_mode const requested_mode = parse_mode(argc, argv);
  auto requirements = vkexec::examples::vulkan_sample_requirements();
  requirements.features.textureCompressionASTC_LDR = VK_TRUE;
  requirements.features.shaderStorageImageExtendedFormats = VK_TRUE;
  auto win = vkexec::examples::glfw_presenter::create({
    .width = k_window_width,
    .height = k_window_height,
    .title = "vkexec Vulkan Samples: Async Compute",
    .validation_layers = true,
    .requirements = std::move(requirements),
  });
  auto &ctx = win.ctx();
  check_image_formats(ctx);
  vkexec::queue_ref const compute_queue = choose_compute_queue(ctx, requested_mode);
  print_queue_mode(ctx, requested_mode, compute_queue);

  auto scene = vkexec::examples::sync_wait_value(
    vkexec::examples::load_gltf_mesh(std::string{ VKEXEC_SAMPLE_ASSET_DIR } + "/bonza/Bonza.gltf", false));
  scene_bounds const bounds = find_bounds(scene.vertices);
  light_data const light = make_light(bounds);
  auto allocator = vkexec::examples::make_vma_allocator(ctx);
  auto vertices = vkexec::examples::sync_wait_value(vkexec::vma::factory::make_gpu_buffer(allocator,
    vkexec::vma::gpu_buffer_create_info{
      .size = std::as_bytes(std::span{ scene.vertices }).size(),
      .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      .memory = vkexec::vma::buffer_memory::host_visible,
    }));
  auto indices = vkexec::examples::sync_wait_value(vkexec::vma::factory::make_gpu_buffer(allocator,
    vkexec::vma::gpu_buffer_create_info{
      .size = std::as_bytes(std::span{ scene.indices }).size(),
      .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      .memory = vkexec::vma::buffer_memory::host_visible,
    }));
  std::memcpy(vertices.mapped().data(), scene.vertices.data(), vertices.mapped().size());
  std::memcpy(indices.mapped().data(), scene.indices.data(), indices.mapped().size());
  check(vertices.flush());
  check(indices.flush());

  auto const material_sampler = vkexec::examples::make_sampler(ctx.device());
  auto const bloom_sampler = vkexec::examples::make_sampler(ctx.device(), VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
  std::filesystem::path const asset_dir{ VKEXEC_SAMPLE_ASSET_DIR };
  std::vector<vkexec::examples::ktx_texture> textures;
  textures.reserve(scene.base_color_textures.size());
  for (std::string const &texture_path : scene.base_color_textures) {
    if (texture_path.empty()) { vkexec::examples::fail_check("Bonza material has no base-color texture"); }
    textures.push_back(vkexec::examples::load_ktx_texture(ctx, allocator, asset_dir / "bonza" / texture_path));
  }
  std::filesystem::path const shader_dir{ VKEXEC_SAMPLE_SHADER_DIR };
  auto compute = make_compute_pipelines(ctx, shader_dir);
  std::vector<frame_resources> frames;
  auto rebuild_frames = [&]() -> void {
    win.wait_idle();
    for (frame_resources &frame : frames) { destroy_frame(ctx, frame); }
    frames.clear();
    frames.reserve(k_frames_in_flight);
    for (std::size_t index = 0; index < k_frames_in_flight; ++index) {
      frames.push_back(make_frame(
        win, allocator, shader_dir, textures, material_sampler, bloom_sampler, compute, light, compute_queue));
    }
  };
  rebuild_frames();
  vkexec::examples::free_camera camera{
    { bounds.center.at(0),
      bounds.center.at(1) + (bounds.radius * k_camera_height_scale),
      bounds.center.at(2) + (bounds.radius * k_camera_distance_scale) },
    { 0.0F, 0.0F, 0.0F, 1.0F },
    bounds.radius * k_camera_speed_scale,
    bounds.radius * k_camera_far_scale,
  };
  std::size_t frame_index = 0;
  bool upload_textures = true;
  while (!win.should_close()) {
    win.poll_events();
    camera.update(win.native_window());
    VkExtent2D const window_extent = win.target().extent();
    if (window_extent.width == 0 || window_extent.height == 0) { continue; }
    VkExtent2D const wanted_extent = scaled_extent(window_extent);
    if (frames.front().hdr.extent().width != wanted_extent.width
        || frames.front().hdr.extent().height != wanted_extent.height) {
      rebuild_frames();
    }
    auto begun = win.begin_frame();
    if (!begun) { vkexec::examples::abort_with_error(begun.error()); }
    if (!begun->has_value()) { continue; }
    vkexec::frame const present_frame = **begun;
    frame_resources &frame = frames.at(frame_index);
    vkexec::examples::camera_data const camera_uniform = camera.data(frame.hdr.extent());
    std::memcpy(frame.camera_buffer.mapped().data(), &camera_uniform, sizeof(camera_uniform));
    check(frame.camera_buffer.flush());
    record_early(ctx, frame, scene, vertices.handle(), indices.handle(), textures, upload_textures, compute_queue);
    record_compute(ctx, frame, compute_queue);
    record_final(ctx, frame, present_frame, win, compute_queue);
    submit_frame(ctx, frame, present_frame, win, compute_queue);
    frame.initialized = true;
    upload_textures = false;
    frame_index = (frame_index + 1) % k_frames_in_flight;
  }
  win.wait_idle();
  for (frame_resources &frame : frames) { destroy_frame(ctx, frame); }
  return 0;
}

}// namespace

auto main(int argc, char const *const *argv) -> int
{
  return vkexec::examples::run_example([argc, argv]() -> int { return run(argc, argv); });
}
