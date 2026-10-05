#include "../common/sampler.hpp"
#include "common/shader_loader.hpp"
#include "common/vulkan_requirements.hpp"
#include "glfw_presenter.hpp"
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
#include <vkexec_extensions/timeline_semaphore/timeline_semaphore.hpp>
#include <vkexec_features/feature.hpp>
#include <vkexec_features/timeline_semaphore.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>
#include <vkexec_graphics/presenter.hpp>
#include <vkexec_vma/allocator.hpp>
#include <vkexec_vma/image.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t k_window_width = 800;
constexpr std::uint32_t k_window_height = 800;
constexpr std::uint32_t k_grid_width = 64;
constexpr std::uint32_t k_grid_height = 64;
constexpr std::uint32_t k_local_size = 8;
constexpr std::uint32_t k_fullscreen_vertices = 3;
constexpr std::size_t k_image_count = 2;
constexpr float k_update_interval_seconds = 1.0F;
constexpr VkFormat k_image_format = VK_FORMAT_R8G8B8A8_UNORM;

enum class timeline_stage : std::uint8_t { submit = 1, draw = 2, present = 3, end = 4 };
constexpr std::uint64_t k_stages_per_frame = static_cast<std::uint64_t>(timeline_stage::end);

[[nodiscard]] constexpr auto timeline_value(std::uint64_t number, timeline_stage stage) -> std::uint64_t
{ return (number * k_stages_per_frame) + static_cast<std::uint64_t>(stage); }

struct game_image
{
  vkexec::vma::image image;
  vkexec::owned::image_view view;
};

struct frame_ticket
{
  std::uint64_t number;
  std::size_t image_index;
  vkexec::frame present_frame;
  vkexec::frame_submit_sync sync;
  float elapsed;
  bool update;
};

struct ticket_exchange
{
  std::mutex mutex;
  std::condition_variable_any changed;
  std::optional<frame_ticket> current;

  auto publish(frame_ticket const &ticket) -> void
  {
    {
      std::scoped_lock const lock{ mutex };
      current = ticket;
    }
    changed.notify_all();
  }

  [[nodiscard]] auto next(std::stop_token const &stopped, std::uint64_t previous) -> std::optional<frame_ticket>
  {
    std::unique_lock lock{ mutex };
    changed.wait(lock, stopped, [&]() -> bool { return current && current->number != previous; });
    if (stopped.stop_requested()) { return std::nullopt; }
    return current;
  }
};

auto check(vkexec::status result) -> void
{
  if (!result) { vkexec::examples::abort_with_error(result.error()); }
}

auto check_vk(VkResult result, char const *message) -> void
{
  if (result != VK_SUCCESS) { vkexec::examples::fail_check(message); }
}

[[nodiscard]] auto make_image(vkexec::context &ctx,
  vkexec::vma::allocator &allocator,
  vkexec::queue_ref compute_queue) -> game_image
{
  std::vector<std::uint32_t> families;
  if (compute_queue.family != ctx.graphics_queue_family()) {
    families = { compute_queue.family, ctx.graphics_queue_family() };
  }
  auto image = vkexec::examples::sync_wait_value(vkexec::allocate_image(allocator,
    vkexec::image_create_info{
      .extent = { .width = k_grid_width, .height = k_grid_height, .depth = 1 },
      .format = k_image_format,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharing_mode = families.empty() ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
      .queue_families = std::move(families),
    }));
  auto view = vkexec::examples::sync_wait_value(vkexec::vma::factory::make_image_view(ctx, image));
  return game_image{ .image = std::move(image), .view = std::move(view) };
}

auto transition(vkexec::context &ctx,
  VkCommandBuffer cmd,
  VkImage image,
  VkImageLayout old_layout,
  VkImageLayout new_layout,
  VkPipelineStageFlags2 src_stage,
  VkPipelineStageFlags2 dst_stage,
  VkAccessFlags2 src_access,
  VkAccessFlags2 dst_access) -> void
{
  check(vkexec::image_barrier(ctx,
    cmd,
    vkexec::image_barrier_params{
      .image = image,
      .old_layout = old_layout,
      .new_layout = new_layout,
      .src_stage = src_stage,
      .dst_stage = dst_stage,
      .src_access = src_access,
      .dst_access = dst_access,
    }));
}

auto begin_recording(VkCommandBuffer cmd) -> void
{
  check_vk(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer failed");
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check_vk(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer failed");
}

[[nodiscard]] auto compute_resources(game_image const &output,
  game_image const &input,
  VkSampler sampler) -> vkexec::resource_table
{
  return vkexec::bindings(
    vkexec::resource_binding{ .slot = 0, .resource = vkexec::storage_image_resource(output.view.handle()) },
    vkexec::resource_binding{ .slot = 1, .resource = vkexec::sampled_image_resource(input.view.handle()) },
    vkexec::resource_binding{ .slot = 2, .resource = vkexec::sampler_resource(sampler) });
}

auto initialize_images(vkexec::context &ctx,
  vkexec::queue_ref compute_queue,
  std::array<game_image, k_image_count> const &images,
  vkexec::owned::compute_pipeline const &pipeline,
  std::array<VkDescriptorSet, k_image_count> const &sets) -> void
{
  auto allocated = ctx.allocate_command_buffer(compute_queue);
  if (!allocated) { vkexec::examples::abort_with_error(allocated.error()); }
  VkCommandBuffer cmd = *allocated;
  begin_recording(cmd);
  constexpr vkexec::dispatch k_groups{
    .x = k_grid_width / k_local_size, .y = k_grid_height / k_local_size, .z = 1 };
  for (std::size_t index = 0; index < k_image_count; ++index) {
    transition(ctx,
      cmd,
      images.at(index).image.handle(),
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_GENERAL,
      VK_PIPELINE_STAGE_2_NONE,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_NONE,
      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkexec::record_pass(cmd, pipeline.bind(sets.at(index)), nullptr, 0, k_groups);
    transition(ctx,
      cmd,
      images.at(index).image.handle(),
      VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  }
  check_vk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer failed for initialization");
  check(ctx.submit_and_wait(cmd, compute_queue));
  ctx.free_command_buffer(cmd);
}

auto record_compute(vkexec::context &ctx,
  VkCommandBuffer cmd,
  frame_ticket const &ticket,
  std::array<game_image, k_image_count> const &images,
  vkexec::owned::compute_pipeline const &update,
  vkexec::owned::compute_pipeline const &mutate,
  std::array<VkDescriptorSet, k_image_count> const &update_sets,
  std::array<VkDescriptorSet, k_image_count> const &mutate_sets) -> void
{
  begin_recording(cmd);
  VkImage output = images.at(ticket.image_index).image.handle();
  transition(ctx,
    cmd,
    output,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_IMAGE_LAYOUT_GENERAL,
    VK_PIPELINE_STAGE_2_NONE,
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    VK_ACCESS_2_NONE,
    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
  constexpr vkexec::dispatch k_groups{
    .x = k_grid_width / k_local_size, .y = k_grid_height / k_local_size, .z = 1 };
  if (ticket.update) {
    vkexec::record_pass(cmd, update.bind(update_sets.at(ticket.image_index)), nullptr, 0, k_groups);
  } else {
    vkexec::record_pass(cmd,
      mutate.bind(mutate_sets.at(ticket.image_index)),
      &ticket.elapsed,
      static_cast<std::uint32_t>(sizeof(ticket.elapsed)),
      k_groups);
  }
  transition(ctx,
    cmd,
    output,
    VK_IMAGE_LAYOUT_GENERAL,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_2_NONE,
    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    VK_ACCESS_2_NONE);
  check_vk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer failed for Game of Life compute");
}

// NOLINTNEXTLINE(bugprone-exception-escape)
auto run() -> int
{
  auto requirements = vkexec::examples::vulkan_sample_requirements();
  vkexec::feat::configure<vkexec::feat::timeline_semaphore>(requirements);
  auto win = vkexec::examples::glfw_presenter::create({
    .width = k_window_width,
    .height = k_window_height,
    .title = "vkexec Vulkan Samples: Timeline Semaphore",
    .validation_layers = true,
    .requirements = std::move(requirements),
  });
  auto &ctx = win.ctx();
  if (!vkexec::feat::available<vkexec::feat::timeline_semaphore>(ctx)) {
    vkexec::examples::fail_check("Timeline Semaphore sample requires the timelineSemaphore Vulkan feature");
  }
  vkexec::queue_ref const compute_queue = ctx.compute_queue_ref();
  auto allocator = vkexec::examples::make_vma_allocator(ctx);
  auto sampler = vkexec::examples::make_sampler(ctx.device(), VK_SAMPLER_ADDRESS_MODE_REPEAT, false, VK_FILTER_NEAREST);
  std::array<game_image, k_image_count> images{
    make_image(ctx, allocator, compute_queue), make_image(ctx, allocator, compute_queue) };

  std::filesystem::path const shader_dir{ VKEXEC_SAMPLE_SHADER_DIR };
  using enum vkexec::buffer_access;
  vkexec::layout_desc const init_layout{
    .binding_kinds = { vkexec::resource_kind::storage_image },
    .binding_slots = {},
    .bindings = { writeonly },
    .push_constant_size = 0,
    .specialization = {},
    .local_size = { k_local_size, k_local_size, 1 },
  };
  vkexec::layout_desc const update_layout{
    .binding_kinds = { vkexec::resource_kind::storage_image,
      vkexec::resource_kind::sampled_image,
      vkexec::resource_kind::sampler },
    .binding_slots = {},
    .bindings = { writeonly, readonly, readonly },
    .push_constant_size = 0,
    .specialization = {},
    .local_size = { k_local_size, k_local_size, 1 },
  };
  auto mutate_layout = update_layout;
  mutate_layout.push_constant_size = sizeof(float);
  auto init = vkexec::examples::sync_wait_value(vkexec::factory::make_compute_pipeline(
    ctx, vkexec::examples::load_spirv(shader_dir / "game_of_life_init.comp.spv"), init_layout));
  auto update = vkexec::examples::sync_wait_value(vkexec::factory::make_compute_pipeline(
    ctx, vkexec::examples::load_spirv(shader_dir / "game_of_life_update.comp.spv"), update_layout));
  auto mutate = vkexec::examples::sync_wait_value(vkexec::factory::make_compute_pipeline(
    ctx, vkexec::examples::load_spirv(shader_dir / "game_of_life_mutate.comp.spv"), mutate_layout));

  std::array<VkDescriptorSet, k_image_count> init_sets{};
  std::array<VkDescriptorSet, k_image_count> update_sets{};
  std::array<VkDescriptorSet, k_image_count> mutate_sets{};
  for (std::size_t index = 0; index < k_image_count; ++index) {
    auto init_set = init.allocate_set();
    auto update_set = update.allocate_set();
    auto mutate_set = mutate.allocate_set();
    if (!init_set || !update_set || !mutate_set) {
      vkexec::examples::fail_check("Game of Life descriptor allocation failed");
    }
    init_sets.at(index) = *init_set;
    update_sets.at(index) = *update_set;
    mutate_sets.at(index) = *mutate_set;
    check(init.update_set(init_sets.at(index), vkexec::bindings(vkexec::resource_binding{
      .slot = 0, .resource = vkexec::storage_image_resource(images.at(index).view.handle()) })));
    std::size_t const input_index = (index + 1) % k_image_count;
    check(update.update_set(update_sets.at(index),
      compute_resources(images.at(index), images.at(input_index), sampler.handle)));
    check(mutate.update_set(mutate_sets.at(index),
      compute_resources(images.at(index), images.at(input_index), sampler.handle)));
  }
  initialize_images(ctx, compute_queue, images, init, init_sets);

  auto vertex_spirv = vkexec::examples::load_spirv(shader_dir / "render.vert.spv");
  auto fragment_spirv = vkexec::examples::load_spirv(shader_dir / "render.frag.spv");
  auto make_renderer = [&](std::size_t index) -> vkexec::owned::graphics_pipeline {
    vkexec::graphics_pipeline_config config{};
    config.clear_r = 0.0F;
    config.clear_g = 0.0F;
    config.clear_b = 0.0F;
    return vkexec::examples::sync_wait_value(vkexec::factory::make_graphics_pipeline(ctx,
      win.render_pass(),
      config,
      vertex_spirv,
      fragment_spirv,
      vkexec::bindings(
        vkexec::resource_binding{
          .slot = 0, .resource = vkexec::sampled_image_resource(images.at(index).view.handle()) },
        vkexec::resource_binding{ .slot = 1, .resource = vkexec::sampler_resource(sampler.handle) })));
  };
  std::array<vkexec::owned::graphics_pipeline, k_image_count> renderers{
    make_renderer(0), make_renderer(1) };
  auto timeline = vkexec::examples::sync_wait_value(vkexec::factory::make_timeline_semaphore(ctx));
  auto command = ctx.allocate_command_buffer(compute_queue);
  if (!command) { vkexec::examples::abort_with_error(command.error()); }
  VkCommandBuffer compute_cmd = *command;
  ticket_exchange tickets;

  std::jthread compute_worker{ [&](std::stop_token const &stopped) -> void {
    std::uint64_t previous = UINT64_MAX;
    while (auto ticket = tickets.next(stopped, previous)) {
      check(timeline.wait(timeline_value(ticket->number, timeline_stage::submit)));
      record_compute(ctx, compute_cmd, *ticket, images, update, mutate, update_sets, mutate_sets);
      std::array<VkCommandBuffer, 1> const commands{ compute_cmd };
      std::array<vkexec::semaphore_submit, 1> const signals{ vkexec::semaphore_submit{
        .semaphore = timeline.handle(), .value = timeline_value(ticket->number, timeline_stage::draw) } };
      check(ctx.submit(vkexec::queue_submit{
        .command_buffers = commands, .signals = signals, .queue = compute_queue.queue }));
      previous = ticket->number;
    }
  } };
  std::jthread graphics_worker{ [&](std::stop_token const &stopped) -> void {
    std::uint64_t previous = UINT64_MAX;
    while (auto ticket = tickets.next(stopped, previous)) {
      check(timeline.wait(timeline_value(ticket->number, timeline_stage::submit)));
      check(renderers.at(ticket->image_index)
          .draw(ticket->present_frame.command_buffer,
            win.render_pass(),
            ticket->present_frame.framebuffer,
            ticket->present_frame.extent,
            k_fullscreen_vertices));
      std::uint64_t const draw_value = timeline_value(ticket->number, timeline_stage::draw);
      if (compute_queue.queue == ctx.graphics_queue()) { check(timeline.wait(draw_value)); }
      std::array<VkCommandBuffer, 1> const commands{ ticket->present_frame.command_buffer };
      std::array<vkexec::semaphore_submit, 2> const waits{
        vkexec::semaphore_submit{
          .semaphore = timeline.handle(), .value = draw_value, .stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT },
        ticket->sync.image_available_wait,
      };
      std::array<vkexec::semaphore_submit, 2> const signals{
        vkexec::semaphore_submit{
          .semaphore = timeline.handle(), .value = timeline_value(ticket->number, timeline_stage::present) },
        ticket->sync.render_finished_signal,
      };
      check(ctx.submit(vkexec::queue_submit{ .command_buffers = commands,
        .waits = waits,
        .signals = signals,
        .fence = ticket->sync.fence,
        .queue = ctx.graphics_queue() }));
      previous = ticket->number;
    }
  } };

  std::cout << "Timeline Semaphore: graphics family " << ctx.graphics_queue_family() << ", compute family "
            << compute_queue.family << "; close the window to exit\n";
  std::uint64_t number = 0;
  auto last_update = std::chrono::steady_clock::now();
  while (!win.should_close()) {
    win.poll_events();
    auto begun = win.begin_frame();
    if (!begun) { vkexec::examples::abort_with_error(begun.error()); }
    if (!begun->has_value()) { continue; }
    auto sync = win.target().submission_sync(**begun);
    if (!sync) { vkexec::examples::abort_with_error(sync.error()); }
    auto const now = std::chrono::steady_clock::now();
    float const elapsed = std::chrono::duration<float>(now - last_update).count();
    bool const do_update = elapsed > k_update_interval_seconds;
    if (do_update) { last_update = now; }
    frame_ticket const ticket{
      .number = number,
      .image_index = number % k_image_count,
      .present_frame = **begun,
      .sync = *sync,
      .elapsed = elapsed,
      .update = do_update,
    };
    tickets.publish(ticket);
    check(timeline.signal(timeline_value(number, timeline_stage::submit)));
    check(timeline.wait(timeline_value(number, timeline_stage::present)));
    auto presented = win.target().present_submitted(ticket.present_frame);
    if (!presented) { vkexec::examples::abort_with_error(presented.error()); }
    check(timeline.signal(timeline_value(number, timeline_stage::end)));
    ++number;
  }
  compute_worker.request_stop();
  graphics_worker.request_stop();
  tickets.changed.notify_all();
  compute_worker.join();
  graphics_worker.join();
  win.wait_idle();
  auto counter = timeline.counter();
  if (!counter) { vkexec::examples::abort_with_error(counter.error()); }
  std::cout << "Final timeline value: " << *counter << '\n';
  ctx.free_command_buffer(compute_cmd);
  return 0;
}

}// namespace

auto main() -> int { return vkexec::examples::run_example(run); }
