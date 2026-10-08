#include "../common/free_camera.hpp"
#include "../common/ktx_texture.hpp"
#include "../common/sampler.hpp"
#include "common/shader_loader.hpp"
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

auto check(vkexec::status result) -> void
{
  if (!result) { vkexec::examples::abort_with_error(result.error()); }
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

[[nodiscard]] constexpr auto color_range() -> VkImageSubresourceRange
{
  return {
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1
  };
}

[[nodiscard]] auto make_frame(vkexec::examples::glfw_presenter &win,
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

auto destroy_frame(vkexec::context &ctx, frame_resources &frame) -> void
{
  for (bloom_pass const &pass : frame.passes) { vkexec::free_compute_set(ctx, pass.pipeline->resources(), pass.set); }
}

[[nodiscard]] constexpr auto depth_range() -> VkImageSubresourceRange
{
  return {
    .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1
  };
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

struct frame_receiver
{
  using receiver_concept = stdexec::receiver_t;

  std::promise<vkexec::status> *completion{ nullptr };

  auto set_value() const && noexcept -> void { completion->set_value(vkexec::status{}); }
  auto set_error(vkexec::error err) const && noexcept -> void { completion->set_value(vkexec::fail(std::move(err))); }
  auto set_error(std::exception_ptr const & /*exception*/) const && noexcept -> void
  { completion->set_value(vkexec::fail(vkexec::errc::unexpected_exception, "frame graph failed with an exception")); }
  auto set_stopped() const && noexcept -> void
  { completion->set_value(vkexec::fail(vkexec::errc::cancelled, "frame graph stopped")); }
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  [[nodiscard]] auto get_env() const noexcept -> stdexec::env<> { return {}; }
};

using frame_sender =
  decltype(std::declval<vkexec::dynamic_pass_graph_sender>()
           | vkexec::present(std::declval<vkexec::owned::presenter &>(), std::declval<vkexec::acquired_frame>()));

struct frame_execution
{
  using operation = decltype(stdexec::connect(std::declval<frame_sender>(), frame_receiver{}));

  std::promise<vkexec::status> completion;
  std::future<vkexec::status> completed;
  operation op;

  explicit frame_execution(frame_sender sender)
    : completed(completion.get_future()), op(stdexec::connect(std::move(sender), frame_receiver{ &completion }))
  { stdexec::start(op); }

  ~frame_execution()
  {
    if (completed.valid()) { completed.wait(); }
  }

  frame_execution(frame_execution const &) = delete;
  auto operator=(frame_execution const &) -> frame_execution & = delete;
  frame_execution(frame_execution &&) = delete;
  auto operator=(frame_execution &&) -> frame_execution & = delete;

  auto wait() -> void
  {
    if (completed.valid()) { check(completed.get()); }
  }
};

using frame_executions = std::array<std::unique_ptr<frame_execution>, k_frames_in_flight>;

auto drain_frames(frame_executions &in_flight) -> void
{
  for (auto &execution : in_flight) {
    if (execution) {
      execution->wait();
      execution.reset();
    }
  }
}

auto rebuild_frames(vkexec::examples::glfw_presenter &win,
  std::vector<frame_resources> &frames,
  frame_executions &in_flight,
  vkexec::vma::allocator &allocator,
  std::filesystem::path const &shader_dir,
  std::vector<vkexec::examples::ktx_texture> const &textures,
  vkexec::examples::sampler_owner const &material_sampler,
  vkexec::examples::sampler_owner const &bloom_sampler,
  compute_pipelines const &compute,
  light_data const &light) -> void
{
  drain_frames(in_flight);
  win.wait_idle();
  for (frame_resources &frame : frames) { destroy_frame(win.ctx(), frame); }
  frames.clear();
  frames.reserve(k_frames_in_flight);
  for (std::size_t index = 0; index < k_frames_in_flight; ++index) {
    frames.push_back(make_frame(win, allocator, shader_dir, textures, material_sampler, bloom_sampler, compute, light));
  }
}

auto render_frame(vkexec::context &ctx,
  frame_resources const &frame,
  vkexec::examples::gltf_mesh_data const &scene,
  VkBuffer vertices,
  VkBuffer indices,
  vkexec::acquired_frame present_frame,
  vkexec::examples::glfw_presenter &win,
  vkexec::queue_ref compute_queue) -> frame_sender
{
  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx.get_scheduler()));
  graph.append(vkexec::on_queue(ctx.graphics_queue_ref()));
  graph.append(vkexec::custom_pass(
    vkexec::uses(vkexec::write(frame.shadow.depth_image(),
                   depth_range(),
                   vkexec::image_usage::depth_attachment,
                   frame.initialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED),
      vkexec::read(vertices, vkexec::buffer_usage::vertex),
      vkexec::read(indices, vkexec::buffer_usage::index)),
    [&frame, &scene, vertices, indices](VkCommandBuffer cmd) -> void {
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
    }));
  graph.append(vkexec::custom_pass(
    vkexec::uses(vkexec::read(frame.shadow.depth_image(), depth_range(), vkexec::image_usage::sampled_fragment),
      vkexec::write(frame.hdr.color_image(0),
        color_range(),
        vkexec::image_usage::color_attachment,
        frame.initialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED),
      vkexec::write(frame.hdr.depth_image(),
        depth_range(),
        vkexec::image_usage::depth_attachment,
        frame.initialized ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED),
      vkexec::read(vertices, vkexec::buffer_usage::vertex),
      vkexec::read(indices, vkexec::buffer_usage::index)),
    [&frame, &scene, vertices, indices](VkCommandBuffer cmd) -> void {
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
    }));
  if (frame.initialized) {
    // Establish the previous frame's graphics ownership before compute writes this image again.
    graph.append(vkexec::custom_pass(vkexec::uses(vkexec::read(frame.bloom.back().image.handle(),
                                       color_range(),
                                       vkexec::image_usage::sampled_fragment,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)),
      [](VkCommandBuffer /*cmd*/) -> void {}));
  }
  graph.append(vkexec::on_queue(compute_queue));
  for (bloom_pass const &pass : frame.passes) {
    bloom_image const &output = frame.bloom.at(pass.output_index);
    bloom_params const params{
      .resolution = { output.extent.width, output.extent.height },
      .inv_resolution = { 1.0F / static_cast<float>(output.extent.width),
        1.0F / static_cast<float>(output.extent.height) },
      .inv_input_resolution = { 1.0F / static_cast<float>(pass.input_extent.width),
        1.0F / static_cast<float>(pass.input_extent.height) },
    };
    graph.append(vkexec::compute_pass(pass.pipeline->bind(pass.set, pass.resources),
      params,
      vkexec::dispatch{ .x = (output.extent.width + k_local_size - 1) / k_local_size,
        .y = (output.extent.height + k_local_size - 1) / k_local_size,
        .z = 1 }));
  }
  graph.append(vkexec::on_queue(ctx.graphics_queue_ref()));
  auto present_use = vkexec::write(present_frame.image, color_range(), vkexec::image_usage::color_attachment);
  present_use.final_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  graph.append(vkexec::custom_pass(
    vkexec::uses(vkexec::read(frame.hdr.color_image(0), color_range(), vkexec::image_usage::sampled_fragment),
      vkexec::read(frame.bloom.back().image.handle(), color_range(), vkexec::image_usage::sampled_fragment),
      present_use),
    [&frame, &win, present_frame](VkCommandBuffer cmd) -> void {
      frame.composite.record_pass(
        cmd, win.render_pass(), present_frame.framebuffer, present_frame.extent, k_fullscreen_vertices);
    }));
  return std::move(graph) | vkexec::present(win.target(), present_frame);
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
  auto uploads = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx.get_scheduler()));
  uploads.append(vkexec::on_queue(ctx.graphics_queue_ref()));
  for (vkexec::examples::ktx_texture const &texture : textures) {
    VkImageSubresourceRange const range{
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .baseMipLevel = 0,
      .levelCount = 1,
      .baseArrayLayer = 0,
      .layerCount = 1,
    };
    uploads.append(
      vkexec::custom_pass(vkexec::uses(vkexec::read(texture.staging.handle(), vkexec::buffer_usage::transfer_source),
                            vkexec::write(texture.image.handle(), range, vkexec::image_usage::transfer_destination)),
        [&ctx, &texture](
          VkCommandBuffer cmd) -> void { vkexec::examples::record_ktx_texture_upload(ctx, cmd, texture); }));
  }
  vkexec::examples::sync_wait_graph(std::move(uploads));
  std::filesystem::path const shader_dir{ VKEXEC_SAMPLE_SHADER_DIR };
  auto compute = make_compute_pipelines(ctx, shader_dir);
  std::vector<frame_resources> frames;
  frame_executions in_flight{};
  rebuild_frames(
    win, frames, in_flight, allocator, shader_dir, textures, material_sampler, bloom_sampler, compute, light);
  vkexec::examples::free_camera camera{
    { bounds.center.at(0),
      bounds.center.at(1) + (bounds.radius * k_camera_height_scale),
      bounds.center.at(2) + (bounds.radius * k_camera_distance_scale) },
    { 0.0F, 0.0F, 0.0F, 1.0F },
    bounds.radius * k_camera_speed_scale,
    bounds.radius * k_camera_far_scale,
  };
  std::size_t frame_index = 0;
  while (!win.should_close()) {
    win.poll_events();
    camera.update(win.native_window());
    VkExtent2D const window_extent = win.target().extent();
    if (window_extent.width == 0 || window_extent.height == 0) { continue; }
    VkExtent2D const wanted_extent = scaled_extent(window_extent);
    if (frames.front().hdr.extent().width != wanted_extent.width
        || frames.front().hdr.extent().height != wanted_extent.height) {
      rebuild_frames(
        win, frames, in_flight, allocator, shader_dir, textures, material_sampler, bloom_sampler, compute, light);
    }
    if (in_flight.at(frame_index)) {
      in_flight.at(frame_index)->wait();
      in_flight.at(frame_index).reset();
    }
    auto acquired = win.target().acquire_frame();
    if (!acquired) { vkexec::examples::abort_with_error(acquired.error()); }
    if (!acquired->has_value()) { continue; }
    vkexec::acquired_frame const present_frame = **acquired;
    frame_resources &frame = frames.at(frame_index);
    vkexec::examples::camera_data const camera_uniform = camera.data(frame.hdr.extent());
    std::memcpy(frame.camera_buffer.mapped().data(), &camera_uniform, sizeof(camera_uniform));
    check(frame.camera_buffer.flush());
    in_flight.at(frame_index) = std::make_unique<frame_execution>(
      render_frame(ctx, frame, scene, vertices.handle(), indices.handle(), present_frame, win, compute_queue));
    frame.initialized = true;
    frame_index = (frame_index + 1) % k_frames_in_flight;
  }
  drain_frames(in_flight);
  win.wait_idle();
  for (frame_resources &frame : frames) { destroy_frame(ctx, frame); }
  return 0;
}

}// namespace

auto main(int argc, char const *const *argv) -> int
{
  return vkexec::examples::run_example([argc, argv]() -> int { return run(argc, argv); });
}
