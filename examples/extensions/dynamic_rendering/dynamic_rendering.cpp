#include "../../glfw_presenter.hpp"
#include <vkexec/barrier.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_extensions/dynamic_rendering/rendering.hpp>
#include <vkexec_features/dynamic_rendering.hpp>
#include <vkexec_features/feature.hpp>
#include <vkexec_graphics/presenter.hpp>
#include <vkexec_graphics/swapchain.hpp>

#include "../../sync_wait_helpers.hpp"

#include <vulkan/vulkan_core.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <utility>

namespace {

constexpr std::uint32_t k_window_width = 800;
constexpr std::uint32_t k_window_height = 600;
constexpr float k_clear_base = 0.35F;
constexpr float k_clear_amplitude = 0.25F;
constexpr float k_phase_step = 0.05F;
constexpr float k_phase_green = 2.094395102F;// 2*pi/3
constexpr float k_phase_blue = 4.188790205F;// 4*pi/3

auto make_requirements() -> vkexec::vulkan_requirements
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 3;
  vkexec::feat::configure<vkexec::feat::dynamic_rendering>(requirements);
  return requirements;
}

auto record_swapchain_clear(vkexec::swapchain const &chain,
  VkCommandBuffer cmd,
  vkexec::frame const &frame,
  float phase) -> vkexec::status
{
  if (frame.image_index >= chain.images().size()) {
    return vkexec::fail(vkexec::errc::out_of_range, "swapchain image index out of range");
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
  VkImage image = chain.images()[frame.image_index];
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
  VkImageView view = chain.image_views()[frame.image_index];

  vkexec::image_barrier(cmd,
    {
      .image = image,
      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
      .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      .dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .src_access = 0,
      .dst_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    });

  vkexec::color_attachment color{};
  color.view = view;
  color.clear.color = {
    { k_clear_base + (k_clear_amplitude * std::sin(phase)),
      k_clear_base + (k_clear_amplitude * std::sin(phase + k_phase_green)),
      k_clear_base + (k_clear_amplitude * std::sin(phase + k_phase_blue)),
      1.0F },
  };
  std::array<vkexec::color_attachment, 1> const colors{ color };

  if (auto began = vkexec::cmd_begin_rendering(cmd,
        vkexec::rendering_info{
          .extent = frame.extent,
          .color = colors,
        });
    !began) {
    return vkexec::fail(std::move(began.error()));
  }
  if (auto ended = vkexec::cmd_end_rendering(cmd); !ended) { return vkexec::fail(std::move(ended.error())); }

  vkexec::image_barrier(cmd,
    {
      .image = image,
      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      .old_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .new_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      .src_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      .src_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .dst_access = 0,
    });
  return {};
}

auto run() -> int
{
  auto win = vkexec::examples::glfw_presenter::create({ .width = k_window_width,
    .height = k_window_height,
    .title = "vkexec dynamic_rendering",
    .requirements = make_requirements() });

  std::cout << std::format("dynamic_rendering: close the window to exit\n");

  float phase = 0.0F;
  while (!win.should_close()) {
    win.poll_events();

    auto frame_result = win.begin_frame();
    if (!frame_result) { vkexec::examples::abort_with_error(frame_result.error()); }
    if (!frame_result->has_value()) { continue; }

    vkexec::frame const frame = **frame_result;
    vkexec::swapchain const *const chain = win.borrowed_swapchain();
    if (chain == nullptr) { vkexec::examples::fail_check("window has no swapchain"); }

    if (auto recorded = record_swapchain_clear(*chain, frame.command_buffer, frame, phase); !recorded) {
      vkexec::examples::abort_with_error(recorded.error());
    }
    if (vkEndCommandBuffer(frame.command_buffer) != VK_SUCCESS) {
      vkexec::examples::fail_check("vkEndCommandBuffer failed");
    }
    if (auto fence = win.end_frame(frame); !fence) { vkexec::examples::abort_with_error(fence.error()); }

    phase += k_phase_step;
  }

  win.wait_idle();
  return 0;
}

}// namespace

auto main() -> int { return vkexec::examples::run_example(run); }
