#include <vkexec/window.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <string>

namespace vkexec {
namespace {

void check(VkResult result, const char *what)
{
  if (result != VK_SUCCESS) { throw std::runtime_error(what); }
}

template<typename T>
T unwrap(vkb::Result<T> result, char const *what)
{
  if (!result) {
    throw std::runtime_error(
      std::string(what) + ": " + result.error().message() + " (" + std::to_string(result.vk_result()) + ")");
  }
  return result.value();
}

} // namespace

void window::on_framebuffer_resize(GLFWwindow *win, int /*width*/, int /*height*/)
{
  auto *self = static_cast<window *>(glfwGetWindowUserPointer(win));
  if (self != nullptr) { self->framebuffer_resized_ = true; }
}

window::window() : window(config{}) {}

window::window(config cfg) : cfg_(std::move(cfg))
{
  if (glfwInit() != GLFW_TRUE) { throw std::runtime_error("glfwInit failed"); }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

  glfw_ = glfwCreateWindow(static_cast<int>(cfg_.width), static_cast<int>(cfg_.height), cfg_.title.c_str(), nullptr,
    nullptr);
  if (glfw_ == nullptr) {
    glfwTerminate();
    throw std::runtime_error("glfwCreateWindow failed");
  }
  glfwSetWindowUserPointer(glfw_, this);
  glfwSetFramebufferSizeCallback(glfw_, &window::on_framebuffer_resize);

  std::uint32_t ext_count = 0;
  const char **glfw_exts = glfwGetRequiredInstanceExtensions(&ext_count);
  if (glfw_exts == nullptr || ext_count == 0) {
    throw std::runtime_error("glfwGetRequiredInstanceExtensions failed (no presentation support?)");
  }
  std::vector<const char *> instance_exts(glfw_exts, glfw_exts + ext_count);

  ctx_ = std::unique_ptr<context>(new context(context::instance_only_tag{}, instance_exts));
  create_surface();
  ctx_->complete_for_surface(surface_);

  create_swapchain();
  create_image_views();
  create_render_pass();
  create_framebuffers();
  create_frame_resources();
}

window::~window()
{
  if (ctx_ && ctx_->device() != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(ctx_->device());
    for (auto &f : frames_) {
      if (f.in_flight != VK_NULL_HANDLE) { vkDestroyFence(ctx_->device(), f.in_flight, nullptr); }
      if (f.render_finished != VK_NULL_HANDLE) { vkDestroySemaphore(ctx_->device(), f.render_finished, nullptr); }
      if (f.image_available != VK_NULL_HANDLE) { vkDestroySemaphore(ctx_->device(), f.image_available, nullptr); }
    }
    if (!command_buffers_.empty()) {
      vkFreeCommandBuffers(ctx_->device(), ctx_->command_pool(), static_cast<std::uint32_t>(command_buffers_.size()),
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

bool window::should_close() const noexcept { return glfwWindowShouldClose(glfw_) == GLFW_TRUE; }

void window::poll_events() { glfwPollEvents(); }

void window::wait_idle()
{
  if (ctx_ && ctx_->device() != VK_NULL_HANDLE) { vkDeviceWaitIdle(ctx_->device()); }
}

void window::create_surface()
{
  check(glfwCreateWindowSurface(ctx_->instance(), glfw_, nullptr, &surface_), "glfwCreateWindowSurface failed");
}

void window::create_swapchain()
{
  int fb_w = 0;
  int fb_h = 0;
  glfwGetFramebufferSize(glfw_, &fb_w, &fb_h);

  vkb::Swapchain old = swapchain_;
  auto builder = vkb::SwapchainBuilder{ ctx_->vkb_device(), surface_ }
                   .set_desired_format({ VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
                   .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                   .set_desired_extent(static_cast<std::uint32_t>(fb_w), static_cast<std::uint32_t>(fb_h))
                   .set_old_swapchain(old);

  swapchain_ = unwrap(builder.build(), "vk-bootstrap SwapchainBuilder");
  if (old.swapchain != VK_NULL_HANDLE) { vkb::destroy_swapchain(old); }

  swapchain_format_ = swapchain_.image_format;
  swapchain_extent_ = swapchain_.extent;
  swapchain_images_ = unwrap(swapchain_.get_images(), "vk-bootstrap Swapchain::get_images");
}

void window::create_image_views()
{
  swapchain_views_ = unwrap(swapchain_.get_image_views(), "vk-bootstrap Swapchain::get_image_views");
}

void window::create_render_pass()
{
  VkAttachmentDescription color{};
  color.format = swapchain_format_;
  color.samples = VK_SAMPLE_COUNT_1_BIT;
  color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference color_ref{};
  color_ref.attachment = 0;
  color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_ref;

  VkSubpassDependency dep{};
  dep.srcSubpass = VK_SUBPASS_EXTERNAL;
  dep.dstSubpass = 0;
  dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dep.srcAccessMask = 0;
  dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo rpci{};
  rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  rpci.attachmentCount = 1;
  rpci.pAttachments = &color;
  rpci.subpassCount = 1;
  rpci.pSubpasses = &subpass;
  rpci.dependencyCount = 1;
  rpci.pDependencies = &dep;
  check(vkCreateRenderPass(ctx_->device(), &rpci, nullptr, &render_pass_), "vkCreateRenderPass failed");
}

void window::create_framebuffers()
{
  framebuffers_.resize(swapchain_views_.size());
  for (std::size_t i = 0; i < swapchain_views_.size(); ++i) {
    VkFramebufferCreateInfo fbci{};
    fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbci.renderPass = render_pass_;
    fbci.attachmentCount = 1;
    fbci.pAttachments = &swapchain_views_[i];
    fbci.width = swapchain_extent_.width;
    fbci.height = swapchain_extent_.height;
    fbci.layers = 1;
    check(vkCreateFramebuffer(ctx_->device(), &fbci, nullptr, &framebuffers_[i]), "vkCreateFramebuffer failed");
  }
}

void window::create_frame_resources()
{
  frames_.resize(k_frames);
  command_buffers_.resize(k_frames);
  for (int i = 0; i < k_frames; ++i) {
    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    check(vkCreateSemaphore(ctx_->device(), &sci, nullptr, &frames_[static_cast<std::size_t>(i)].image_available),
      "vkCreateSemaphore failed");
    check(vkCreateSemaphore(ctx_->device(), &sci, nullptr, &frames_[static_cast<std::size_t>(i)].render_finished),
      "vkCreateSemaphore failed");

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    check(vkCreateFence(ctx_->device(), &fci, nullptr, &frames_[static_cast<std::size_t>(i)].in_flight),
      "vkCreateFence failed");

    command_buffers_[static_cast<std::size_t>(i)] = ctx_->allocate_command_buffer();
  }
}

void window::cleanup_swapchain()
{
  for (VkFramebuffer fb : framebuffers_) {
    if (fb != VK_NULL_HANDLE) { vkDestroyFramebuffer(ctx_->device(), fb, nullptr); }
  }
  framebuffers_.clear();
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

void window::recreate_swapchain()
{
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(glfw_, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(glfw_, &width, &height);
    glfwWaitEvents();
  }

  vkDeviceWaitIdle(ctx_->device());

  for (VkFramebuffer fb : framebuffers_) {
    if (fb != VK_NULL_HANDLE) { vkDestroyFramebuffer(ctx_->device(), fb, nullptr); }
  }
  framebuffers_.clear();
  if (!swapchain_views_.empty()) {
    swapchain_.destroy_image_views(swapchain_views_);
    swapchain_views_.clear();
  }
  swapchain_images_.clear();

  create_swapchain();
  create_image_views();
  create_framebuffers();
}

std::optional<frame> window::begin_frame()
{
  if (frame_open_) { throw std::logic_error("begin_frame called while a frame is already open"); }

  auto &sync = frames_[frame_index_];
  check(vkWaitForFences(ctx_->device(), 1, &sync.in_flight, VK_TRUE, UINT64_MAX), "vkWaitForFences failed");

  std::uint32_t image_index = 0;
  VkResult acquire = vkAcquireNextImageKHR(ctx_->device(), swapchain_, UINT64_MAX, sync.image_available, VK_NULL_HANDLE,
    &image_index);
  if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
    recreate_swapchain();
    return std::nullopt;
  }
  if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("vkAcquireNextImageKHR failed");
  }

  check(vkResetFences(ctx_->device(), 1, &sync.in_flight), "vkResetFences failed");

  VkCommandBuffer cmd = command_buffers_[frame_index_];
  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer failed");

  current_image_index_ = image_index;
  frame_open_ = true;
  return frame{ cmd, framebuffers_[image_index], swapchain_extent_, image_index };
}

void window::end_frame(const frame &drawn)
{
  if (!frame_open_) { throw std::logic_error("end_frame called without begin_frame"); }
  (void)drawn;

  auto &sync = frames_[frame_index_];
  VkCommandBuffer cmd = command_buffers_[frame_index_];

  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.waitSemaphoreCount = 1;
  submit.pWaitSemaphores = &sync.image_available;
  submit.pWaitDstStageMask = &wait_stage;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  submit.signalSemaphoreCount = 1;
  submit.pSignalSemaphores = &sync.render_finished;
  check(vkQueueSubmit(ctx_->graphics_queue(), 1, &submit, sync.in_flight), "vkQueueSubmit failed");

  VkPresentInfoKHR present{};
  present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present.waitSemaphoreCount = 1;
  present.pWaitSemaphores = &sync.render_finished;
  present.swapchainCount = 1;
  present.pSwapchains = &swapchain_.swapchain;
  present.pImageIndices = &current_image_index_;

  VkResult presented = vkQueuePresentKHR(ctx_->present_queue(), &present);
  if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR || framebuffer_resized_) {
    framebuffer_resized_ = false;
    recreate_swapchain();
  } else if (presented != VK_SUCCESS) {
    throw std::runtime_error("vkQueuePresentKHR failed");
  }

  frame_index_ = (frame_index_ + 1) % static_cast<std::uint32_t>(k_frames);
  frame_open_ = false;
}

} // namespace vkexec
