#include "common/shader_loader.hpp"
#include "common/vulkan_requirements.hpp"
#include "glfw_presenter.hpp"
#include "load_gltf_mesh.hpp"
#include "sponza_texture.hpp"
#include "sync_wait_helpers.hpp"

#include <vkexec/barrier.hpp>
#include <vkexec/context.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec_features/feature.hpp>
#include <vkexec_features/shader_demote_to_helper_invocation.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>
#include <vkexec_graphics/presenter.hpp>
#include <vkexec_vma/allocator.hpp>
#include <vkexec_vma/gpu_buffer.hpp>
#include <vkexec_vma/offscreen_target.hpp>

#include <GLFW/glfw3.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
constexpr std::uint32_t k_window_width = 800;
constexpr std::uint32_t k_window_height = 600;
constexpr std::uint32_t k_lighting_vertices = 3;
constexpr std::size_t k_frames_in_flight = 2;
constexpr std::uint32_t k_gbuffer_color_count = 2;
constexpr VkFormat k_color_format = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat k_depth_format = VK_FORMAT_D32_SFLOAT;
constexpr float k_camera_speed = 250.0F;
constexpr float k_mouse_sensitivity = 0.002F;
constexpr float k_camera_max_pitch = 1.55F;
constexpr float k_camera_far_plane = 4000.0F;
constexpr std::array<float, 3> k_camera_initial_position{ -705.01001F, 195.20282F, -119.932266F };
constexpr float k_camera_rotation_x = -0.004728F;
constexpr float k_camera_rotation_y = -0.775409F;
constexpr float k_camera_rotation_z = -0.005807F;
constexpr float k_camera_rotation_w = 0.631416F;

enum class barrier_mode : std::uint8_t { fragment, conservative };

struct sampler_owner
{
  VkDevice device{ VK_NULL_HANDLE };
  VkSampler handle{ VK_NULL_HANDLE };
  sampler_owner(VkDevice owned_device, VkSampler owned_handle) noexcept : device(owned_device), handle(owned_handle) {}
  sampler_owner(sampler_owner const &) = delete;
  auto operator=(sampler_owner const &) -> sampler_owner & = delete;
  sampler_owner(sampler_owner &&) = delete;
  auto operator=(sampler_owner &&) -> sampler_owner & = delete;
  ~sampler_owner()
  {
    if (handle != VK_NULL_HANDLE) { vkDestroySampler(device, handle, nullptr); }
  }
};

struct frame_resources
{
  vkexec::vma::offscreen_target target;
  vkexec::vma::gpu_buffer camera_buffer;
  std::vector<vkexec::owned::graphics_pipeline> geometry;
  vkexec::owned::graphics_pipeline lighting;
  bool initialized{ false };
};

struct camera_data
{
  std::array<float, 4> position{};
  std::array<float, 4> right{};
  std::array<float, 4> up{};
  std::array<float, 4> forward{};
  std::array<float, 4> projection{};
};

struct free_camera
{
  std::array<float, 3> position = k_camera_initial_position;
  float yaw{ 0.0F };
  float pitch{ 0.0F };
  double previous_time{ 0.0 };
  double previous_x{ 0.0 };
  double previous_y{ 0.0 };
  bool dragging{ false };

  free_camera()
  {
    // Main Camera rotation from Sponza01.gltf.
    float const forward_x =
      -2.0F * ((k_camera_rotation_x * k_camera_rotation_z) + (k_camera_rotation_w * k_camera_rotation_y));
    float const forward_y =
      -2.0F * ((k_camera_rotation_y * k_camera_rotation_z) - (k_camera_rotation_w * k_camera_rotation_x));
    float const forward_z =
      -(1.0F - (2.0F * ((k_camera_rotation_x * k_camera_rotation_x) + (k_camera_rotation_y * k_camera_rotation_y))));
    yaw = std::atan2(forward_x, -forward_z);
    pitch = std::asin(std::clamp(forward_y, -1.0F, 1.0F));
  }

  auto update(GLFWwindow *window) -> void
  {
    double const now = glfwGetTime();
    float const delta = static_cast<float>(std::clamp(now - previous_time, 0.0, 0.1));
    previous_time = now;
    double cursor_x = 0.0;
    double cursor_y = 0.0;
    glfwGetCursorPos(window, &cursor_x, &cursor_y);
    bool const pressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (pressed && dragging) {
      yaw += static_cast<float>(cursor_x - previous_x) * k_mouse_sensitivity;
      pitch = std::clamp(pitch - (static_cast<float>(cursor_y - previous_y) * k_mouse_sensitivity),
        -k_camera_max_pitch,
        k_camera_max_pitch);
    }
    dragging = pressed;
    previous_x = cursor_x;
    previous_y = cursor_y;
    float const sine = std::sin(yaw);
    float const cosine = std::cos(yaw);
    std::array<float, 3> const forward{ std::cos(pitch) * sine, std::sin(pitch), -std::cos(pitch) * cosine };
    std::array<float, 3> const right{ cosine, 0.0F, sine };
    auto move = [&](int key, std::array<float, 3> const &direction, float sign) -> void {
      if (glfwGetKey(window, key) != GLFW_PRESS) { return; }
      for (std::size_t index = 0; index < position.size(); ++index) {
        position.at(index) += direction.at(index) * sign * k_camera_speed * delta;
      }
    };
    move(GLFW_KEY_W, forward, 1.0F);
    move(GLFW_KEY_S, forward, -1.0F);
    move(GLFW_KEY_D, right, 1.0F);
    move(GLFW_KEY_A, right, -1.0F);
    move(GLFW_KEY_E, { 0.0F, 1.0F, 0.0F }, 1.0F);
    move(GLFW_KEY_Q, { 0.0F, 1.0F, 0.0F }, -1.0F);
  }

  [[nodiscard]] auto data(VkExtent2D extent) const -> camera_data
  {
    float const sine = std::sin(yaw);
    float const cosine = std::cos(yaw);
    float const pitch_sine = std::sin(pitch);
    float const pitch_cosine = std::cos(pitch);
    return camera_data{
      .position = { position.at(0), position.at(1), position.at(2), 0.0F },
      .right = { cosine, 0.0F, sine, 0.0F },
      .up = { -pitch_sine * sine, pitch_cosine, pitch_sine * cosine, 0.0F },
      .forward = { pitch_cosine * sine, pitch_sine, -pitch_cosine * cosine, 0.0F },
      .projection = { 1.0F,
        k_camera_far_plane,
        static_cast<float>(extent.width) / static_cast<float>(extent.height),
        0.0F },
    };
  }
};

auto make_sampler(VkDevice device) -> VkSampler
{
  VkSamplerCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  info.magFilter = VK_FILTER_LINEAR;
  info.minFilter = VK_FILTER_LINEAR;
  info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  info.maxLod = 1.0F;
  VkSampler sampler{ VK_NULL_HANDLE };
  if (vkCreateSampler(device, &info, nullptr, &sampler) != VK_SUCCESS) {
    vkexec::examples::fail_check("vkCreateSampler failed");
  }
  return sampler;
}

auto make_frame_resources(vkexec::examples::glfw_presenter &win,
  vkexec::vma::allocator &allocator,
  std::filesystem::path const &shader_dir,
  std::vector<vkexec::examples::sponza_texture> const &textures,
  vkexec::vma::gpu_buffer camera_buffer,
  VkSampler sampler,
  bool albedo_only) -> frame_resources
{
  auto target_result = vkexec::vma::offscreen_target::create(win.ctx(),
    allocator,
    vkexec::vma::offscreen_target_config{
      .extent = win.target().extent(),
      .color_formats = { k_color_format, k_color_format },
      .depth_format = k_depth_format,
      .sample_depth = true,
    });
  if (!target_result) { vkexec::examples::abort_with_error(target_result.error()); }
  auto target = std::move(*target_result);

  vkexec::graphics_pipeline_config geometry_config{};
  geometry_config.depth_test = true;
  geometry_config.use_mesh_vertices = true;
  geometry_config.color_attachment_count = k_gbuffer_color_count;
  auto const geometry_vertex_shader = vkexec::examples::load_spirv(shader_dir / "geometry.vert.spv");
  auto const geometry_fragment_shader = vkexec::examples::load_spirv(shader_dir / "geometry.frag.spv");
  std::vector<vkexec::owned::graphics_pipeline> geometry;
  geometry.reserve(textures.size());
  for (vkexec::examples::sponza_texture const &texture : textures) {
    auto material_resources = vkexec::bindings(
      vkexec::resource_binding{ .slot = 0, .resource = vkexec::sampled_image_resource(texture.view.handle()) },
      vkexec::resource_binding{ .slot = 1, .resource = vkexec::sampler_resource(sampler) },
      vkexec::resource_binding{
        .slot = 2, .resource = vkexec::buffer_resource(camera_buffer.handle(), sizeof(camera_data)) });
    geometry.push_back(vkexec::examples::sync_wait_value(vkexec::factory::make_graphics_pipeline(win.ctx(),
      target.render_pass(),
      geometry_config,
      geometry_vertex_shader,
      geometry_fragment_shader,
      std::move(material_resources))));
  }

  auto resources = vkexec::bindings(
    vkexec::resource_binding{ .slot = 0, .resource = vkexec::sampled_image_resource(target.color_view(0)) },
    vkexec::resource_binding{ .slot = 1, .resource = vkexec::sampled_image_resource(target.color_view(1)) },
    vkexec::resource_binding{ .slot = 2, .resource = vkexec::sampled_image_resource(target.depth_view()) },
    vkexec::resource_binding{ .slot = 3, .resource = vkexec::sampler_resource(sampler) });
  auto lighting = vkexec::examples::sync_wait_value(vkexec::factory::make_graphics_pipeline(win.ctx(),
    win.render_pass(),
    vkexec::graphics_pipeline_config{},
    vkexec::examples::load_spirv(shader_dir / "lighting.vert.spv"),
    vkexec::examples::load_spirv(shader_dir / (albedo_only ? "albedo.frag.spv" : "lighting.frag.spv")),
    std::move(resources)));
  return frame_resources{ .target = std::move(target),
    .camera_buffer = std::move(camera_buffer),
    .geometry = std::move(geometry),
    .lighting = std::move(lighting) };
}

auto transition_image(vkexec::context &ctx,
  VkCommandBuffer cmd,
  VkImage image,
  VkImageAspectFlags aspect,
  VkImageLayout old_layout,
  VkImageLayout new_layout,
  VkPipelineStageFlags2 source_stage,
  VkPipelineStageFlags2 destination_stage,
  VkAccessFlags2 source_access,
  VkAccessFlags2 destination_access) -> void
{
  auto barrier = vkexec::image_barrier(ctx,
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
    });
  if (!barrier) { vkexec::examples::abort_with_error(barrier.error()); }
}

auto prepare_geometry(vkexec::context &ctx, VkCommandBuffer cmd, frame_resources const &frame) -> void
{
  VkImageLayout const old_layout =
    frame.initialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
  VkPipelineStageFlags2 const source_stage =
    frame.initialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE;
  VkAccessFlags2 const source_access = frame.initialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE;
  for (std::size_t index = 0; index < k_gbuffer_color_count; ++index) {
    transition_image(ctx,
      cmd,
      frame.target.color_image(index),
      VK_IMAGE_ASPECT_COLOR_BIT,
      old_layout,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      source_stage,
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      source_access,
      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  }
  transition_image(ctx,
    cmd,
    frame.target.depth_image(),
    VK_IMAGE_ASPECT_DEPTH_BIT,
    old_layout,
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    source_stage,
    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
    source_access,
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
}

auto finish_geometry(vkexec::context &ctx, VkCommandBuffer cmd, frame_resources const &frame, barrier_mode mode) -> void
{
  VkPipelineStageFlags2 const color_source = mode == barrier_mode::fragment
                                               ? VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
                                               : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  VkPipelineStageFlags2 const depth_source =
    mode == barrier_mode::fragment
      ? VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
      : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  VkPipelineStageFlags2 const destination =
    mode == barrier_mode::fragment ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  for (std::size_t index = 0; index < k_gbuffer_color_count; ++index) {
    transition_image(ctx,
      cmd,
      frame.target.color_image(index),
      VK_IMAGE_ASPECT_COLOR_BIT,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      color_source,
      destination,
      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  }
  transition_image(ctx,
    cmd,
    frame.target.depth_image(),
    VK_IMAGE_ASPECT_DEPTH_BIT,
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    depth_source,
    destination,
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

auto record_frame(vkexec::context &ctx,
  vkexec::frame const &present_frame,
  vkexec::examples::glfw_presenter &win,
  frame_resources &resources,
  vkexec::examples::gltf_mesh_data const &scene_data,
  VkBuffer vertex_buffer,
  VkBuffer index_buffer,
  std::vector<vkexec::examples::sponza_texture> const &textures,
  bool upload_textures,
  barrier_mode mode) -> void
{
  VkCommandBuffer cmd = present_frame.command_buffer;
  if (upload_textures) {
    for (vkexec::examples::sponza_texture const &texture : textures) {
      vkexec::examples::record_sponza_texture_upload(ctx, cmd, texture);
    }
  }
  prepare_geometry(ctx, cmd, resources);
  auto const clears = resources.target.clear_values();
  VkRenderPassBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  begin.renderPass = resources.target.render_pass();
  begin.framebuffer = resources.target.framebuffer();
  begin.renderArea = VkRect2D{ .offset = VkOffset2D{ .x = 0, .y = 0 }, .extent = resources.target.extent() };
  begin.clearValueCount = static_cast<std::uint32_t>(clears.size());
  begin.pClearValues = clears.data();
  vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
  for (vkexec::examples::gltf_mesh_data::primitive_draw const &draw : scene_data.draws) {
    if (draw.material_index < 0 || std::cmp_greater_equal(draw.material_index, resources.geometry.size())) {
      vkexec::examples::fail_check("Sponza primitive has no base-color material");
    }
    resources.geometry.at(static_cast<std::size_t>(draw.material_index))
      .record_draw(cmd,
        resources.target.extent(),
        vkexec::mesh_draw{
          .vertex_buffer = vertex_buffer,
          .index_buffer = index_buffer,
          .index_count = draw.index_count,
          .first_index = draw.first_index,
        });
  }
  vkCmdEndRenderPass(cmd);
  finish_geometry(ctx, cmd, resources, mode);
  resources.initialized = true;
  if (auto drawn = resources.lighting.draw(
        cmd, win.render_pass(), present_frame.framebuffer, present_frame.extent, k_lighting_vertices);
    !drawn) {
    vkexec::examples::abort_with_error(drawn.error());
  }
}

auto parse_mode(int argc, char const *const *argv) -> barrier_mode
{
  barrier_mode mode = barrier_mode::fragment;
  for (int index = 1; index < argc; ++index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::string_view const argument(argv[index]);
    if (argument == "--barrier=conservative") {
      mode = barrier_mode::conservative;
    } else if (argument != "--barrier=fragment" && argument != "--view=albedo") {
      vkexec::examples::fail_check("use --barrier=fragment, --barrier=conservative, or --view=albedo");
    }
  }
  return mode;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
auto run(int argc, char const *const *argv) -> int
{
  barrier_mode const mode = parse_mode(argc, argv);
  bool albedo_only = false;
  for (int index = 1; index < argc; ++index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    albedo_only = albedo_only || std::string_view(argv[index]) == "--view=albedo";
  }
  auto requirements = vkexec::examples::vulkan_sample_requirements();
  requirements.features.textureCompressionASTC_LDR = VK_TRUE;
  vkexec::feat::configure<vkexec::feat::shader_demote_to_helper_invocation>(requirements);
  auto win = vkexec::examples::glfw_presenter::create({
    .width = k_window_width,
    .height = k_window_height,
    .title = "vkexec Vulkan Samples: Pipeline Barriers",
    .validation_layers = true,
    .requirements = std::move(requirements),
  });
  auto &ctx = win.ctx();
  VkFormatProperties texture_format_properties{};
  vkGetPhysicalDeviceFormatProperties(ctx.physical_device(), VK_FORMAT_ASTC_8x8_SRGB_BLOCK, &texture_format_properties);
  constexpr VkFormatFeatureFlags k_texture_format_features =
    VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
  if ((texture_format_properties.optimalTilingFeatures & k_texture_format_features) != k_texture_format_features) {
    vkexec::examples::fail_check("Sponza requires sampled ASTC 8x8 sRGB textures on this device");
  }
  auto scene_data = vkexec::examples::sync_wait_value(
    vkexec::examples::load_gltf_mesh(std::string{ VKEXEC_SAMPLE_ASSET_DIR } + "/sponza/Sponza01.gltf", false));
  auto allocator = vkexec::examples::make_vma_allocator(ctx);
  auto vertices = vkexec::examples::sync_wait_value(vkexec::vma::factory::make_gpu_buffer(allocator,
    vkexec::vma::gpu_buffer_create_info{
      .size = std::as_bytes(std::span{ scene_data.vertices }).size(),
      .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      .memory = vkexec::vma::buffer_memory::host_visible,
    }));
  auto indices = vkexec::examples::sync_wait_value(vkexec::vma::factory::make_gpu_buffer(allocator,
    vkexec::vma::gpu_buffer_create_info{
      .size = std::as_bytes(std::span{ scene_data.indices }).size(),
      .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      .memory = vkexec::vma::buffer_memory::host_visible,
    }));
  std::memcpy(vertices.mapped().data(), scene_data.vertices.data(), vertices.mapped().size());
  std::memcpy(indices.mapped().data(), scene_data.indices.data(), indices.mapped().size());
  if (auto flushed = vertices.flush(); !flushed) { vkexec::examples::abort_with_error(flushed.error()); }
  if (auto flushed = indices.flush(); !flushed) { vkexec::examples::abort_with_error(flushed.error()); }
  VkSampler sampler_handle = make_sampler(ctx.device());
  sampler_owner const sampler{ ctx.device(), sampler_handle };
  std::filesystem::path const asset_dir{ VKEXEC_SAMPLE_ASSET_DIR };
  std::vector<vkexec::examples::sponza_texture> textures;
  textures.reserve(scene_data.base_color_textures.size());
  for (std::string const &texture_path : scene_data.base_color_textures) {
    if (texture_path.empty()) { vkexec::examples::fail_check("Sponza material is missing a base-color texture"); }
    textures.push_back(vkexec::examples::load_sponza_texture(ctx, allocator, asset_dir / "sponza" / texture_path));
  }
  std::filesystem::path const shader_dir{ VKEXEC_SAMPLE_SHADER_DIR };
  std::vector<frame_resources> frames;
  auto rebuild_frames = [&]() -> void {
    win.wait_idle();
    frames.clear();
    frames.reserve(k_frames_in_flight);
    for (std::size_t index = 0; index < k_frames_in_flight; ++index) {
      auto camera_buffer = vkexec::examples::sync_wait_value(vkexec::vma::factory::make_gpu_buffer(allocator,
        vkexec::vma::gpu_buffer_create_info{
          .size = sizeof(camera_data),
          .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          .memory = vkexec::vma::buffer_memory::host_visible,
        }));
      frames.push_back(make_frame_resources(
        win, allocator, shader_dir, textures, std::move(camera_buffer), sampler.handle, albedo_only));
    }
  };
  rebuild_frames();

  std::size_t frame_index = 0;
  bool upload_textures = true;
  free_camera camera;
  std::cout << "Pipeline Barriers: " << (mode == barrier_mode::fragment ? "fragment" : "conservative")
            << " synchronization; " << (albedo_only ? "albedo view" : "lit view")
            << "; WASD move, Q/E descend/ascend, right mouse drag to look; close the window to exit\n";
  while (!win.should_close()) {
    win.poll_events();
    camera.update(win.native_window());
    if (frames.front().target.extent().width != win.target().extent().width
        || frames.front().target.extent().height != win.target().extent().height) {
      rebuild_frames();
    }
    auto begun = win.begin_frame();
    if (!begun) { vkexec::examples::abort_with_error(begun.error()); }
    if (!begun->has_value()) { continue; }
    vkexec::frame const present_frame = **begun;
    camera_data const camera_uniform = camera.data(present_frame.extent);
    auto &camera_buffer = frames.at(frame_index).camera_buffer;
    std::memcpy(camera_buffer.mapped().data(), &camera_uniform, sizeof(camera_uniform));
    if (auto flushed = camera_buffer.flush(); !flushed) { vkexec::examples::abort_with_error(flushed.error()); }
    record_frame(ctx,
      present_frame,
      win,
      frames.at(frame_index),
      scene_data,
      vertices.handle(),
      indices.handle(),
      textures,
      upload_textures,
      mode);
    upload_textures = false;
    auto ended = win.end_frame(present_frame);
    if (!ended) { vkexec::examples::abort_with_error(ended.error()); }
    frame_index = (frame_index + 1) % k_frames_in_flight;
  }
  win.wait_idle();
  return 0;
}
}// namespace

auto main(int argc, char const *const *argv) -> int
{
  return vkexec::examples::run_example([argc, argv]() -> int { return run(argc, argv); });
}
