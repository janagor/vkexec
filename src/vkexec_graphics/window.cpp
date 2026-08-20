#include <vkexec_graphics/window.hpp>

#include <vkexec/context.hpp>
#include <vkexec/detail/config.hpp>

#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <vulkan/vulkan_core.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vkexec {
namespace {

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  std::string g_glfw_error;

  auto glfw_error_callback(int code, char const *description) -> void
  {
    g_glfw_error = "GLFW " + std::to_string(code) + ": " + (description != nullptr ? description : "(no description)");
  }

  constexpr std::uint32_t k_color_attachment_index = 0;
  constexpr std::uint32_t k_depth_attachment_index = 1;
  constexpr std::uint32_t k_framebuffer_attachment_count = 2;
  constexpr std::uint32_t k_depth_mip_levels = 1;
  constexpr std::uint32_t k_depth_array_layers = 1;

  auto check(VkResult result, char const *what) -> void
  {
    if (result != VK_SUCCESS) { VKEXEC_THROW(std::runtime_error(what)); }
  }

  auto pick_depth_format(VkPhysicalDevice phys) -> VkFormat
  {
    static constexpr std::array k_candidates{
      VK_FORMAT_D32_SFLOAT,
      VK_FORMAT_D32_SFLOAT_S8_UINT,
      VK_FORMAT_D24_UNORM_S8_UINT,
    };
    for (VkFormat const format : k_candidates) {
      VkFormatProperties properties{};
      vkGetPhysicalDeviceFormatProperties(phys, format, &properties);
      if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0U) { return format; }
    }
    VKEXEC_THROW(std::runtime_error("no supported depth format"));
  }

  auto format_has_stencil(VkFormat format) -> bool
  { return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT; }

  auto depth_aspect_mask(VkFormat format) -> VkImageAspectFlags
  {
    if (format_has_stencil(format)) {
      return static_cast<VkImageAspectFlags>(static_cast<std::uint32_t>(VK_IMAGE_ASPECT_DEPTH_BIT)
                                             | static_cast<std::uint32_t>(VK_IMAGE_ASPECT_STENCIL_BIT));
    }
    return VK_IMAGE_ASPECT_DEPTH_BIT;
  }

  template<typename T> auto unwrap(vkb::Result<T> result, char const *what) -> T
  {
    if (!result) {
      VKEXEC_THROW(std::runtime_error(
        std::string(what) + ": " + result.error().message() + " (" + std::to_string(result.vk_result()) + ")"));
    }
    return result.value();
  }

}// namespace

// GLFW callback signature is fixed (two adjacent int parameters).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto window::on_framebuffer_resize(GLFWwindow *win, int width, int height) -> void
{
  (void)width;
  (void)height;
  auto *self = static_cast<window *>(glfwGetWindowUserPointer(win));
  if (self != nullptr) { self->framebuffer_resized_ = true; }
}

window::window() : window(config{}) {}

window::window(config cfg) : cfg_(std::move(cfg))
{
  g_glfw_error.clear();
  glfwSetErrorCallback(glfw_error_callback);
  if (glfwInit() != GLFW_TRUE) {
    VKEXEC_THROW(
      std::runtime_error(g_glfw_error.empty() ? "glfwInit failed" : ("glfwInit failed (" + g_glfw_error + ")")));
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

  glfw_ =
    glfwCreateWindow(static_cast<int>(cfg_.width), static_cast<int>(cfg_.height), cfg_.title.c_str(), nullptr, nullptr);
  if (glfw_ == nullptr) {
    glfwTerminate();
    VKEXEC_THROW(std::runtime_error("glfwCreateWindow failed"));
  }
  glfwSetWindowUserPointer(glfw_, this);
  glfwSetFramebufferSizeCallback(glfw_, &window::on_framebuffer_resize);

  std::uint32_t ext_count = 0;
  char const *const *glfw_exts = glfwGetRequiredInstanceExtensions(&ext_count);
  if (glfw_exts == nullptr || ext_count == 0) {
    VKEXEC_THROW(std::runtime_error("glfwGetRequiredInstanceExtensions failed (no presentation support?)"));
  }
  std::vector<char const *> instance_exts;
  instance_exts.reserve(ext_count);
  for (std::uint32_t index = 0; index < ext_count; ++index) {
    instance_exts.push_back(glfw_exts[index]);// NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  }

  ctx_ = std::unique_ptr<context>(new context(
    context::instance_only_tag{}, scheduler_options{ .validation_layers = cfg_.validation_layers }, instance_exts));
  create_surface();
  ctx_->complete_for_surface(surface_);

  create_swapchain();
  create_image_views();
  create_render_pass();
  create_depth_resources();
  create_framebuffers();
  create_frame_resources();
}

window::~window()
{
  if (ctx_ && ctx_->device() != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(ctx_->device());
    for (auto &sync : frames_) {
      if (sync.in_flight != VK_NULL_HANDLE) { vkDestroyFence(ctx_->device(), sync.in_flight, nullptr); }
      if (sync.image_available != VK_NULL_HANDLE) { vkDestroySemaphore(ctx_->device(), sync.image_available, nullptr); }
    }
    destroy_swapchain_sync();
    if (!command_buffers_.empty()) {
      vkFreeCommandBuffers(ctx_->device(),
        ctx_->command_pool(),
        static_cast<std::uint32_t>(command_buffers_.size()),
        command_buffers_.data());
    }
    cleanup_swapchain();
    if (render_pass_ != VK_NULL_HANDLE) { vkDestroyRenderPass(ctx_->device(), render_pass_, nullptr); }
    if (surface_ != VK_NULL_HANDLE) {
      vkb::destroy_surface(ctx_->instance(), surface_);
      surface_ = VK_NULL_HANDLE;
    }
  }
  ctx_.reset();
  if (glfw_ != nullptr) {
    glfwDestroyWindow(glfw_);
    glfw_ = nullptr;
  }
  glfwTerminate();
}

auto window::should_close() const noexcept -> bool { return glfwWindowShouldClose(glfw_) == GLFW_TRUE; }

// cppcheck-suppress functionStatic
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto window::poll_events() -> void { glfwPollEvents(); }

auto window::wait_idle() -> void
{
  if (ctx_ && ctx_->device() != VK_NULL_HANDLE) { vkDeviceWaitIdle(ctx_->device()); }
}

auto window::create_surface() -> void
{ check(glfwCreateWindowSurface(ctx_->instance(), glfw_, nullptr, &surface_), "glfwCreateWindowSurface failed"); }

auto window::create_swapchain() -> void
{
  int framebuffer_width = 0;
  int framebuffer_height = 0;
  glfwGetFramebufferSize(glfw_, &framebuffer_width, &framebuffer_height);

  vkb::Swapchain const old_swapchain = swapchain_;
  auto builder =
    vkb::SwapchainBuilder{ ctx_->vkb_device(), surface_ }
      .set_desired_format({ .format = VK_FORMAT_B8G8R8A8_SRGB, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
      .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
      .set_desired_extent(static_cast<std::uint32_t>(framebuffer_width), static_cast<std::uint32_t>(framebuffer_height))
      .set_old_swapchain(old_swapchain);

  swapchain_ = unwrap(builder.build(), "vk-bootstrap SwapchainBuilder");
  if (old_swapchain.swapchain != VK_NULL_HANDLE) { vkb::destroy_swapchain(old_swapchain); }

  swapchain_format_ = swapchain_.image_format;
  swapchain_extent_ = swapchain_.extent;
  swapchain_images_ = unwrap(swapchain_.get_images(), "vk-bootstrap Swapchain::get_images");
  create_swapchain_sync();
}

auto window::create_image_views() -> void
{ swapchain_views_ = unwrap(swapchain_.get_image_views(), "vk-bootstrap Swapchain::get_image_views"); }

auto window::create_render_pass() -> void
{
  if (depth_format_ == VK_FORMAT_UNDEFINED) { depth_format_ = pick_depth_format(ctx_->physical_device()); }

  VkAttachmentDescription const color{
    .flags = 0,
    .format = swapchain_format_,
    .samples = VK_SAMPLE_COUNT_1_BIT,
    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
  };

  bool const has_stencil = format_has_stencil(depth_format_);
  VkAttachmentDescription const depth{
    .flags = 0,
    .format = depth_format_,
    .samples = VK_SAMPLE_COUNT_1_BIT,
    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
    .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    .stencilLoadOp = has_stencil ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
  };

  std::array const attachments{ color, depth };

  VkAttachmentReference const color_ref{
    .attachment = k_color_attachment_index,
    .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
  };
  VkAttachmentReference const depth_ref{
    .attachment = k_depth_attachment_index,
    .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
  };

  VkSubpassDescription const subpass{
    .flags = 0,
    .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
    .inputAttachmentCount = 0,
    .pInputAttachments = nullptr,
    .colorAttachmentCount = 1,
    .pColorAttachments = &color_ref,
    .pResolveAttachments = nullptr,
    .pDepthStencilAttachment = &depth_ref,
    .preserveAttachmentCount = 0,
    .pPreserveAttachments = nullptr,
  };

  auto const attachment_stages =
    static_cast<VkPipelineStageFlags>(static_cast<std::uint32_t>(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
                                      | static_cast<std::uint32_t>(VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT)
                                      | static_cast<std::uint32_t>(VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT));
  auto const attachment_access =
    static_cast<VkAccessFlags>(static_cast<std::uint32_t>(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
                               | static_cast<std::uint32_t>(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT));

  VkSubpassDependency const dependency{
    .srcSubpass = VK_SUBPASS_EXTERNAL,
    .dstSubpass = 0,
    .srcStageMask = attachment_stages,
    .dstStageMask = attachment_stages,
    .srcAccessMask = 0,
    .dstAccessMask = attachment_access,
    .dependencyFlags = 0,
  };

  VkRenderPassCreateInfo const render_pass_info{
    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .attachmentCount = static_cast<std::uint32_t>(attachments.size()),
    .pAttachments = attachments.data(),
    .subpassCount = 1,
    .pSubpasses = &subpass,
    .dependencyCount = 1,
    .pDependencies = &dependency,
  };
  check(vkCreateRenderPass(ctx_->device(), &render_pass_info, nullptr, &render_pass_), "vkCreateRenderPass failed");
}

auto window::create_depth_resources() -> void
{
  // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = depth_format_;
  image_info.extent = { .width = swapchain_extent_.width, .height = swapchain_extent_.height, .depth = 1 };
  image_info.mipLevels = k_depth_mip_levels;
  image_info.arrayLayers = k_depth_array_layers;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  check(vmaCreateImage(ctx_->allocator(), &image_info, &alloc_info, &depth_image_, &depth_allocation_, nullptr),
    "vmaCreateImage failed (depth)");

  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = depth_image_;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = depth_format_;
  view_info.subresourceRange.aspectMask = depth_aspect_mask(depth_format_);
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = k_depth_mip_levels;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = k_depth_array_layers;
  check(vkCreateImageView(ctx_->device(), &view_info, nullptr, &depth_view_), "vkCreateImageView failed (depth)");
}

auto window::destroy_depth_resources() noexcept -> void
{
  if (ctx_ == nullptr || ctx_->device() == VK_NULL_HANDLE) { return; }
  if (depth_view_ != VK_NULL_HANDLE) {
    vkDestroyImageView(ctx_->device(), depth_view_, nullptr);
    depth_view_ = VK_NULL_HANDLE;
  }
  if (depth_image_ != VK_NULL_HANDLE) {
    vmaDestroyImage(ctx_->allocator(), depth_image_, depth_allocation_);
    depth_image_ = VK_NULL_HANDLE;
    depth_allocation_ = VK_NULL_HANDLE;
  }
}

auto window::create_framebuffers() -> void
{
  framebuffers_.resize(swapchain_views_.size());
  for (std::size_t index = 0; index < swapchain_views_.size(); ++index) {
    std::array const views{ swapchain_views_.at(index), depth_view_ };
    VkFramebufferCreateInfo framebuffer_info{};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = render_pass_;
    framebuffer_info.attachmentCount = k_framebuffer_attachment_count;
    framebuffer_info.pAttachments = views.data();
    framebuffer_info.width = swapchain_extent_.width;
    framebuffer_info.height = swapchain_extent_.height;
    framebuffer_info.layers = 1;
    check(vkCreateFramebuffer(ctx_->device(), &framebuffer_info, nullptr, &framebuffers_.at(index)),
      "vkCreateFramebuffer failed");
  }
}

auto window::create_frame_resources() -> void
{
  frames_.resize(static_cast<std::size_t>(k_frames));
  command_buffers_.resize(static_cast<std::size_t>(k_frames));
  for (int index = 0; index < k_frames; ++index) {
    auto const frame_index = static_cast<std::size_t>(index);
    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    check(vkCreateSemaphore(ctx_->device(), &semaphore_info, nullptr, &frames_.at(frame_index).image_available),
      "vkCreateSemaphore failed");

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    check(
      vkCreateFence(ctx_->device(), &fence_info, nullptr, &frames_.at(frame_index).in_flight), "vkCreateFence failed");

    command_buffers_.at(frame_index) = ctx_->allocate_command_buffer();
  }
}

auto window::create_swapchain_sync() -> void
{
  destroy_swapchain_sync();

  render_finished_.resize(swapchain_images_.size());
  images_in_flight_.assign(swapchain_images_.size(), VK_NULL_HANDLE);

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  for (VkSemaphore &render_finished : render_finished_) {
    check(vkCreateSemaphore(ctx_->device(), &semaphore_info, nullptr, &render_finished),
      "vkCreateSemaphore failed (render_finished)");
  }
}

auto window::destroy_swapchain_sync() noexcept -> void
{
  if (ctx_ == nullptr || ctx_->device() == VK_NULL_HANDLE) {
    render_finished_.clear();
    images_in_flight_.clear();
    return;
  }
  for (VkSemaphore render_finished : render_finished_) {
    if (render_finished != VK_NULL_HANDLE) { vkDestroySemaphore(ctx_->device(), render_finished, nullptr); }
  }
  render_finished_.clear();
  images_in_flight_.clear();
}

auto window::cleanup_swapchain() -> void
{
  destroy_swapchain_sync();
  for (VkFramebuffer framebuffer : framebuffers_) {
    if (framebuffer != VK_NULL_HANDLE) { vkDestroyFramebuffer(ctx_->device(), framebuffer, nullptr); }
  }
  framebuffers_.clear();
  destroy_depth_resources();
  if (!swapchain_views_.empty()) {
    swapchain_.destroy_image_views(swapchain_views_);
    swapchain_views_.clear();
  }
  swapchain_images_.clear();
  if (swapchain_.swapchain != VK_NULL_HANDLE) {
    vkb::destroy_swapchain(swapchain_);
    swapchain_ = {};
  }
}

auto window::recreate_swapchain() -> void
{
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(glfw_, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(glfw_, &width, &height);
    glfwWaitEvents();
  }

  vkDeviceWaitIdle(ctx_->device());

  for (VkFramebuffer framebuffer : framebuffers_) {
    if (framebuffer != VK_NULL_HANDLE) { vkDestroyFramebuffer(ctx_->device(), framebuffer, nullptr); }
  }
  framebuffers_.clear();
  destroy_depth_resources();
  if (!swapchain_views_.empty()) {
    swapchain_.destroy_image_views(swapchain_views_);
    swapchain_views_.clear();
  }
  swapchain_images_.clear();

  create_swapchain();
  create_image_views();
  create_depth_resources();
  create_framebuffers();
}

auto window::begin_frame() -> std::optional<frame>
{
  if (frame_open_) { VKEXEC_THROW(std::logic_error("begin_frame called while a frame is already open")); }

  auto &sync = frames_.at(frame_index_);
  check(vkWaitForFences(ctx_->device(), 1, &sync.in_flight, VK_TRUE, UINT64_MAX), "vkWaitForFences failed");

  std::uint32_t image_index = 0;
  VkResult const acquire =
    vkAcquireNextImageKHR(ctx_->device(), swapchain_, UINT64_MAX, sync.image_available, VK_NULL_HANDLE, &image_index);
  if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
    recreate_swapchain();
    return std::nullopt;
  }
  if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) {
    VKEXEC_THROW(std::runtime_error("vkAcquireNextImageKHR failed"));
  }

  if (images_in_flight_.at(image_index) != VK_NULL_HANDLE) {
    check(vkWaitForFences(ctx_->device(), 1, &images_in_flight_.at(image_index), VK_TRUE, UINT64_MAX),
      "vkWaitForFences failed (swapchain image in flight)");
  }
  images_in_flight_.at(image_index) = sync.in_flight;

  check(vkResetFences(ctx_->device(), 1, &sync.in_flight), "vkResetFences failed");

  VkCommandBuffer cmd = command_buffers_.at(frame_index_);
  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  check(vkBeginCommandBuffer(cmd, &begin_info), "vkBeginCommandBuffer failed");

  current_image_index_ = image_index;
  frame_open_ = true;
  return frame{ .command_buffer = cmd,
    .framebuffer = framebuffers_.at(image_index),
    .extent = swapchain_extent_,
    .image_index = image_index };
}

auto window::end_frame(frame const &drawn) -> void
{
  if (!frame_open_) { VKEXEC_THROW(std::logic_error("end_frame called without begin_frame")); }
  (void)drawn;

  auto &sync = frames_.at(frame_index_);
  VkCommandBuffer cmd = command_buffers_.at(frame_index_);

  VkPipelineStageFlags const wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.waitSemaphoreCount = 1;
  submit_info.pWaitSemaphores = &sync.image_available;
  submit_info.pWaitDstStageMask = &wait_stage;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cmd;
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = &render_finished_.at(current_image_index_);
  check(vkQueueSubmit(ctx_->graphics_queue(), 1, &submit_info, sync.in_flight), "vkQueueSubmit failed");

  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = &render_finished_.at(current_image_index_);
  present_info.swapchainCount = 1;
  present_info.pSwapchains = &swapchain_.swapchain;
  present_info.pImageIndices = &current_image_index_;

  VkResult const presented = vkQueuePresentKHR(ctx_->present_queue(), &present_info);
  if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR || framebuffer_resized_) {
    framebuffer_resized_ = false;
    recreate_swapchain();
  } else if (presented != VK_SUCCESS) {
    VKEXEC_THROW(std::runtime_error("vkQueuePresentKHR failed"));
  }

  frame_index_ = (frame_index_ + 1) % static_cast<std::uint32_t>(k_frames);
  frame_open_ = false;
}

}// namespace vkexec
