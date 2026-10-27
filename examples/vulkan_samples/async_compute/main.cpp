#include <array>
#include <cstring>
#include <exception>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexec/execution.hpp>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/result.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>
#include <vkexec_graphics/present.hpp>
#include <vkexec_graphics/presenter.hpp>
#include <vkexec_vma/gpu_buffer.hpp>
#include <vkexec_vma/image.hpp>
#include <vkexec_vma/offscreen_target.hpp>
#include <vulkan/vulkan_core.h>

#include "../common/free_camera.hpp"
#include "../common/ktx_texture.hpp"
#include "../common/sampler.hpp"
#include "common/smoke_frames.hpp"
#include "common/vulkan_requirements.hpp"
#include "glfw_presenter.hpp"
#include "load_gltf_mesh.hpp"
#include "scene.hpp"
#include "sync_wait_helpers.hpp"
#include "vkexec/detail/pass_types.hpp"
#include "vkexec/vulkan_requirements.hpp"


namespace vkexec::vma {
class allocator;
}// namespace vkexec::vma


namespace vkexec::examples::async_compute {
namespace {

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
    // cppcheck-suppress functionStatic
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

  auto drain_frames(frame_executions &in_flight, vkexec::examples::smoke_frames *smoke = nullptr) -> void
  {
    for (auto &execution : in_flight) {
      if (execution) {
        execution->wait();
        if (smoke != nullptr) { smoke->complete(); }
        execution.reset();
      }
    }
  }

  auto rebuild_frames(vkexec::examples::glfw_presenter &win,
    std::vector<frame_resources> &frames,
    frame_executions &in_flight,
    vkexec::examples::smoke_frames *smoke,
    vkexec::vma::allocator &allocator,
    std::filesystem::path const &shader_dir,
    std::vector<vkexec::examples::ktx_texture> const &textures,
    vkexec::examples::sampler_owner const &material_sampler,
    vkexec::examples::sampler_owner const &bloom_sampler,
    compute_pipelines const &compute,
    light_data const &light) -> void
  {
    drain_frames(in_flight, smoke);
    win.wait_idle();
    for (frame_resources &frame : frames) { destroy_frame(win.ctx(), frame); }
    frames.clear();
    frames.reserve(k_frames_in_flight);
    for (std::size_t index = 0; index < k_frames_in_flight; ++index) {
      frames.push_back(
        make_frame(win, allocator, shader_dir, textures, material_sampler, bloom_sampler, compute, light));
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
      } else if (vkexec::examples::smoke_frames::is_frame_argument(argument)
                 || vkexec::examples::smoke_frames::is_smoke_asset_argument(argument)
                 || argument == "--expect-queue=shared" || argument == "--expect-queue=dedicated") {
        continue;
      } else {
        vkexec::examples::fail_check("use --queues=single or --queues=async");
      }
    }
    return mode;
  }

  auto check_image_formats(vkexec::context &ctx, bool uncompressed_smoke) -> void
  {
    if (!uncompressed_smoke) {
      VkFormatProperties texture_properties{};
      vkGetPhysicalDeviceFormatProperties(ctx.physical_device(), VK_FORMAT_ASTC_5x5_SRGB_BLOCK, &texture_properties);
      constexpr VkFormatFeatureFlags k_texture_features =
        static_cast<VkFormatFeatureFlags>(VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
        | static_cast<VkFormatFeatureFlags>(VK_FORMAT_FEATURE_TRANSFER_DST_BIT);
      if ((texture_properties.optimalTilingFeatures & k_texture_features) != k_texture_features) {
        vkexec::examples::fail_check("Bonza requires sampled ASTC 5x5 sRGB textures");
      }
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

  auto check_queue_expectations(int argc,
    char const *const *argv,
    queue_mode requested_mode,
    vkexec::context &ctx,
    vkexec::queue_ref compute_queue) -> void
  {
    bool expect_shared = false;
    bool expect_dedicated = false;
    for (int index = 1; index < argc; ++index) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      std::string_view const argument(argv[index]);
      expect_shared = expect_shared || argument == "--expect-queue=shared";
      expect_dedicated = expect_dedicated || argument == "--expect-queue=dedicated";
    }
    if (expect_shared && expect_dedicated) { vkexec::examples::fail_check("conflicting queue expectations"); }
    bool const shared_queue = compute_queue.family == ctx.graphics_queue_family();
    if (requested_mode == queue_mode::single && !shared_queue) {
      vkexec::examples::fail_check("single queue mode selected a distinct compute family");
    }
    if (expect_shared && !shared_queue) { vkexec::examples::fail_check("expected shared compute queue"); }
    if (expect_dedicated && shared_queue) { vkexec::examples::fail_check("expected dedicated compute queue"); }
  }

  // NOLINTNEXTLINE(bugprone-exception-escape)
  auto run(int argc, char const *const *argv) -> int
  {
    auto frames_to_run = vkexec::examples::smoke_frames::parse(argc, argv);
    queue_mode const requested_mode = parse_mode(argc, argv);
    auto requirements = vkexec::examples::vulkan_sample_requirements();
    if (!frames_to_run.uncompressed()) { requirements.features.textureCompressionASTC_LDR = VK_TRUE; }
    requirements.features.shaderStorageImageExtendedFormats = VK_TRUE;
    auto win = vkexec::examples::glfw_presenter::create({
      .width = k_window_width,
      .height = k_window_height,
      .title = "vkexec Vulkan Samples: Async Compute",
      .validation_layers = true,
      .requirements = std::move(requirements),
    });
    auto &ctx = win.ctx();
    check_image_formats(ctx, frames_to_run.uncompressed());
    vkexec::queue_ref const compute_queue = choose_compute_queue(ctx, requested_mode);
    check_queue_expectations(argc, argv, requested_mode, ctx, compute_queue);
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
      textures.push_back(vkexec::examples::load_ktx_texture(
        ctx, allocator, asset_dir / "bonza" / texture_path, frames_to_run.uncompressed()));
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
    rebuild_frames(win,
      frames,
      in_flight,
      &frames_to_run,
      allocator,
      shader_dir,
      textures,
      material_sampler,
      bloom_sampler,
      compute,
      light);
    vkexec::examples::free_camera camera{
      { bounds.center.at(0),
        bounds.center.at(1) + (bounds.radius * k_camera_height_scale),
        bounds.center.at(2) + (bounds.radius * k_camera_distance_scale) },
      { 0.0F, 0.0F, 0.0F, 1.0F },
      bounds.radius * k_camera_speed_scale,
      bounds.radius * k_camera_far_scale,
    };
    std::size_t frame_index = 0;
    while (!win.should_close() && frames_to_run.should_continue()) {
      win.poll_events();
      if (!frames_to_run.automated()) { camera.update(win.native_window()); }
      VkExtent2D const window_extent = win.target().extent();
      if (window_extent.width == 0 || window_extent.height == 0) { continue; }
      VkExtent2D const wanted_extent = scaled_extent(window_extent);
      if (frames.front().hdr.extent().width != wanted_extent.width
          || frames.front().hdr.extent().height != wanted_extent.height) {
        rebuild_frames(win,
          frames,
          in_flight,
          &frames_to_run,
          allocator,
          shader_dir,
          textures,
          material_sampler,
          bloom_sampler,
          compute,
          light);
      }
      if (in_flight.at(frame_index)) {
        in_flight.at(frame_index)->wait();
        frames_to_run.complete();
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
      frames_to_run.schedule();
      frame.initialized = true;
      frame_index = (frame_index + 1) % k_frames_in_flight;
    }
    drain_frames(in_flight, &frames_to_run);
    win.wait_idle();
    for (frame_resources &frame : frames) { destroy_frame(ctx, frame); }
    frames_to_run.verify_complete();
    return 0;
  }

}// namespace
}// namespace vkexec::examples::async_compute

auto main(int argc, char const *const *argv) -> int
{
  return vkexec::examples::run_example(
    [argc, argv]() -> int { return vkexec::examples::async_compute::run(argc, argv); });
}
