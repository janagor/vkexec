#include <vkexec_graphics/window.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/vulkan_requirements.hpp>

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

  auto pick_depth_format(VkPhysicalDevice phys) -> result<VkFormat>
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
    return make_error(errc::unsupported, "no supported depth format");
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

auto window::create(config cfg) -> result<window>
{
  window created;
  if (auto init = created.init(std::move(cfg)); !init) { return init.error(); }
  return created;
}

auto window::headless() -> result<window> { return headless(config{}); }

auto window::headless(config cfg) -> result<window>
{
  cfg.headless = true;
  return create(std::move(cfg));
}

window::window(window &&other) noexcept
  : cfg_(std::move(other.cfg_)), headless_(other.headless_), glfw_(other.glfw_), ctx_(std::move(other.ctx_)),
    surface_(other.surface_), swapchain_(std::move(other.swapchain_)), depth_format_(other.depth_format_),
    framebuffers_(std::move(other.framebuffers_)), depth_image_(other.depth_image_),
    depth_allocation_(other.depth_allocation_), depth_view_(other.depth_view_), render_pass_(other.render_pass_),
    frames_(std::move(other.frames_)), command_buffers_(std::move(other.command_buffers_)),
    render_finished_(std::move(other.render_finished_)), images_in_flight_(std::move(other.images_in_flight_)),
    frame_index_(other.frame_index_), current_image_index_(other.current_image_index_),
    framebuffer_resized_(other.framebuffer_resized_), frame_open_(other.frame_open_)
{
  other.glfw_ = nullptr;
  other.surface_ = VK_NULL_HANDLE;
  other.depth_image_ = VK_NULL_HANDLE;
  other.depth_allocation_ = VK_NULL_HANDLE;
  other.depth_view_ = VK_NULL_HANDLE;
  other.render_pass_ = VK_NULL_HANDLE;
  if (glfw_ != nullptr) { glfwSetWindowUserPointer(glfw_, this); }
}

auto window::operator=(window &&other) noexcept -> window &
{
  if (this == &other) { return *this; }
  this->~window();
  new (this) window(std::move(other));
  return *this;
}

auto window::init(config cfg) -> status
{
  cfg_ = std::move(cfg);
  headless_ = cfg_.headless;

  if (headless_) {
    auto const surface_exts = vulkan_library::required_headless_surface_instance_extensions();
    ctx_ = std::unique_ptr<context>(new context(context::instance_only_tag{},
      scheduler_options{ .validation_layers = cfg_.validation_layers, .requirements = cfg_.requirements },
      std::vector<char const *>{ surface_exts.begin(), surface_exts.end() }));
    if (auto surface = create_headless_surface(); !surface) { return surface.error(); }
  } else {
    g_glfw_error.clear();
    glfwSetErrorCallback(glfw_error_callback);
    if (glfwInit() != GLFW_TRUE) {
      return make_error(
        errc::io_error, g_glfw_error.empty() ? "glfwInit failed" : ("glfwInit failed (" + g_glfw_error + ")"));
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    glfw_ = glfwCreateWindow(
      static_cast<int>(cfg_.width), static_cast<int>(cfg_.height), cfg_.title.c_str(), nullptr, nullptr);
    if (glfw_ == nullptr) {
      glfwTerminate();
      return make_error(errc::io_error, "glfwCreateWindow failed");
    }
    glfwSetWindowUserPointer(glfw_, this);
    glfwSetFramebufferSizeCallback(glfw_, &window::on_framebuffer_resize);

    std::uint32_t ext_count = 0;
    char const *const *glfw_exts = glfwGetRequiredInstanceExtensions(&ext_count);
    if (glfw_exts == nullptr || ext_count == 0) {
      return make_error(errc::unsupported, "glfwGetRequiredInstanceExtensions failed (no presentation support?)");
    }
    std::vector<char const *> instance_exts;
    instance_exts.reserve(ext_count);
    for (std::uint32_t index = 0; index < ext_count; ++index) {
      instance_exts.push_back(glfw_exts[index]);// NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    ctx_ = std::unique_ptr<context>(new context(context::instance_only_tag{},
      scheduler_options{ .validation_layers = cfg_.validation_layers, .requirements = cfg_.requirements },
      instance_exts));
    if (auto surface = create_surface(); !surface) { return surface.error(); }
  }

  if (auto completed = ctx_->complete_for_surface(surface_); !completed) { return completed.error(); }
  if (auto swapchain = create_swapchain(); !swapchain) { return swapchain.error(); }
  if (auto render_pass = create_render_pass(); !render_pass) { return render_pass.error(); }
  if (auto depth = create_depth_resources(); !depth) { return depth.error(); }
  if (auto framebuffers = create_framebuffers(); !framebuffers) { return framebuffers.error(); }
  if (auto frames = create_frame_resources(); !frames) { return frames.error(); }
  return {};
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
    glfwTerminate();
  }
}

auto window::should_close() const noexcept -> bool
{
  if (headless_) { return false; }
  return glfwWindowShouldClose(glfw_) == GLFW_TRUE;
}

// cppcheck-suppress functionStatic
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
auto window::poll_events() const -> void
{
  if (!headless_) { glfwPollEvents(); }
}

auto window::wait_idle() -> void
{
  if (ctx_ && ctx_->device() != VK_NULL_HANDLE) { vkDeviceWaitIdle(ctx_->device()); }
}

auto window::create_surface() -> status
{
  if (VkResult const result = glfwCreateWindowSurface(ctx_->instance(), glfw_, nullptr, &surface_);
    result != VK_SUCCESS) {
    return make_vk_error(result, "glfwCreateWindowSurface failed");
  }
  return {};
}

auto window::create_headless_surface() -> status
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto const create_fn = reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(
    vkGetInstanceProcAddr(ctx_->instance(), "vkCreateHeadlessSurfaceEXT"));
  if (create_fn == nullptr) { return make_error(errc::unsupported, "vkCreateHeadlessSurfaceEXT not available"); }

  VkHeadlessSurfaceCreateInfoEXT create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
  if (VkResult const result = create_fn(ctx_->instance(), &create_info, nullptr, &surface_); result != VK_SUCCESS) {
    return make_vk_error(result, "vkCreateHeadlessSurfaceEXT failed");
  }
  return {};
}

auto window::framebuffer_size() const -> std::pair<std::uint32_t, std::uint32_t>
{
  if (headless_) { return { cfg_.width, cfg_.height }; }
  int framebuffer_width = 0;
  int framebuffer_height = 0;
  glfwGetFramebufferSize(glfw_, &framebuffer_width, &framebuffer_height);
  return { static_cast<std::uint32_t>(framebuffer_width), static_cast<std::uint32_t>(framebuffer_height) };
}

auto window::create_swapchain() -> status
{
  auto const [framebuffer_width, framebuffer_height] = framebuffer_size();
  if (!swapchain_.has_value()) {
    auto created = swapchain::create(*ctx_,
      swapchain_create_info{
        .surface = surface_,
        .width = framebuffer_width,
        .height = framebuffer_height,
      });
    if (!created) { return created.error(); }
    swapchain_ = std::move(*created);
  } else if (auto recreated = swapchain_->recreate(framebuffer_width, framebuffer_height); !recreated) {
    return recreated.error();
  }
  return create_swapchain_sync();
}

auto window::create_render_pass() -> status
{
  if (depth_format_ == VK_FORMAT_UNDEFINED) {
    auto format = pick_depth_format(ctx_->physical_device());
    if (!format) { return format.error(); }
    depth_format_ = *format;
  }

  VkAttachmentDescription const color{
    .flags = 0,
    .format = swapchain_format(),
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
  if (VkResult const result = vkCreateRenderPass(ctx_->device(), &render_pass_info, nullptr, &render_pass_);
    result != VK_SUCCESS) {
    return make_vk_error(result, "vkCreateRenderPass failed");
  }
  return {};
}

auto window::create_depth_resources() -> status
{
  // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = depth_format_;
  image_info.extent = { .width = extent().width, .height = extent().height, .depth = 1 };
  image_info.mipLevels = k_depth_mip_levels;
  image_info.arrayLayers = k_depth_array_layers;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  if (VkResult const result =
        vmaCreateImage(ctx_->allocator(), &image_info, &alloc_info, &depth_image_, &depth_allocation_, nullptr);
    result != VK_SUCCESS) {
    return make_vk_error(result, "vmaCreateImage failed (depth)");
  }

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
  if (VkResult const result = vkCreateImageView(ctx_->device(), &view_info, nullptr, &depth_view_);
    result != VK_SUCCESS) {
    return make_vk_error(result, "vkCreateImageView failed (depth)");
  }
  return {};
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

auto window::create_framebuffers() -> status
{
  auto const views = swapchain_->image_views();
  framebuffers_.resize(views.size());
  for (std::size_t index = 0; index < views.size(); ++index) {
    std::array const attachments{ views[index], depth_view_ };
    VkFramebufferCreateInfo framebuffer_info{};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = render_pass_;
    framebuffer_info.attachmentCount = k_framebuffer_attachment_count;
    framebuffer_info.pAttachments = attachments.data();
    framebuffer_info.width = extent().width;
    framebuffer_info.height = extent().height;
    framebuffer_info.layers = 1;
    if (VkResult const result =
          vkCreateFramebuffer(ctx_->device(), &framebuffer_info, nullptr, &framebuffers_.at(index));
      result != VK_SUCCESS) {
      return make_vk_error(result, "vkCreateFramebuffer failed");
    }
  }
  return {};
}

auto window::create_frame_resources() -> status
{
  frames_.resize(static_cast<std::size_t>(k_frames));
  command_buffers_.resize(static_cast<std::size_t>(k_frames));
  for (int index = 0; index < k_frames; ++index) {
    auto const frame_index = static_cast<std::size_t>(index);
    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    if (VkResult const result =
          vkCreateSemaphore(ctx_->device(), &semaphore_info, nullptr, &frames_.at(frame_index).image_available);
      result != VK_SUCCESS) {
      return make_vk_error(result, "vkCreateSemaphore failed");
    }

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (VkResult const result = vkCreateFence(ctx_->device(), &fence_info, nullptr, &frames_.at(frame_index).in_flight);
      result != VK_SUCCESS) {
      return make_vk_error(result, "vkCreateFence failed");
    }

    auto cmd = ctx_->allocate_command_buffer();
    if (!cmd) { return cmd.error(); }
    command_buffers_.at(frame_index) = *cmd;
  }
  return {};
}

auto window::create_swapchain_sync() -> status
{
  destroy_swapchain_sync();

  auto const image_count = swapchain_->images().size();
  render_finished_.resize(image_count);
  images_in_flight_.assign(image_count, VK_NULL_HANDLE);

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  for (VkSemaphore &render_finished : render_finished_) {
    if (VkResult const result = vkCreateSemaphore(ctx_->device(), &semaphore_info, nullptr, &render_finished);
      result != VK_SUCCESS) {
      return make_vk_error(result, "vkCreateSemaphore failed (render_finished)");
    }
  }
  return {};
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
  swapchain_.reset();
}

auto window::recreate_swapchain() -> status
{
  if (!headless_) {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(glfw_, &width, &height);
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(glfw_, &width, &height);
      glfwWaitEvents();
    }
  }

  vkDeviceWaitIdle(ctx_->device());

  for (VkFramebuffer framebuffer : framebuffers_) {
    if (framebuffer != VK_NULL_HANDLE) { vkDestroyFramebuffer(ctx_->device(), framebuffer, nullptr); }
  }
  framebuffers_.clear();
  destroy_depth_resources();

  if (auto swapchain = create_swapchain(); !swapchain) { return swapchain.error(); }
  if (auto depth = create_depth_resources(); !depth) { return depth.error(); }
  if (auto framebuffers = create_framebuffers(); !framebuffers) { return framebuffers.error(); }
  return {};
}

auto window::begin_frame() -> result<std::optional<frame>>
{
  if (frame_open_) { return make_error(errc::invalid_argument, "begin_frame called while a frame is already open"); }

  auto &sync = frames_.at(frame_index_);
  if (VkResult const wait_result = vkWaitForFences(ctx_->device(), 1, &sync.in_flight, VK_TRUE, UINT64_MAX);
    wait_result != VK_SUCCESS) {
    return make_vk_error(wait_result, "vkWaitForFences failed");
  }

  auto acquired = swapchain_->acquire_next_image(sync.image_available);
  if (!acquired) { return acquired.error(); }
  if (!acquired->has_value()) {
    if (auto recreated = recreate_swapchain(); !recreated) { return recreated.error(); }
    return std::optional<frame>{};
  }
  std::uint32_t const image_index = **acquired;

  if (images_in_flight_.at(image_index) != VK_NULL_HANDLE) {
    if (VkResult const wait_result =
          vkWaitForFences(ctx_->device(), 1, &images_in_flight_.at(image_index), VK_TRUE, UINT64_MAX);
      wait_result != VK_SUCCESS) {
      return make_vk_error(wait_result, "vkWaitForFences failed (swapchain image in flight)");
    }
  }
  images_in_flight_.at(image_index) = sync.in_flight;

  if (VkResult const reset_result = vkResetFences(ctx_->device(), 1, &sync.in_flight); reset_result != VK_SUCCESS) {
    return make_vk_error(reset_result, "vkResetFences failed");
  }

  VkCommandBuffer cmd = command_buffers_.at(frame_index_);
  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (VkResult const begin_result = vkBeginCommandBuffer(cmd, &begin_info); begin_result != VK_SUCCESS) {
    return make_vk_error(begin_result, "vkBeginCommandBuffer failed");
  }

  current_image_index_ = image_index;
  frame_open_ = true;
  return frame{
    .command_buffer = cmd, .framebuffer = framebuffers_.at(image_index), .extent = extent(), .image_index = image_index
  };
}

auto window::end_frame(frame const &drawn) -> result<VkFence>
{
  if (!frame_open_) { return make_error(errc::invalid_argument, "end_frame called without begin_frame"); }
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
  if (VkResult const submit_result = vkQueueSubmit(ctx_->graphics_queue(), 1, &submit_info, sync.in_flight);
    submit_result != VK_SUCCESS) {
    return make_vk_error(submit_result, "vkQueueSubmit failed");
  }

  std::array<VkSemaphore, 1> const wait_semaphores{ render_finished_.at(current_image_index_) };
  auto present_result = swapchain_->present(current_image_index_, wait_semaphores);
  if (!present_result) { return present_result.error(); }
  bool const needs_recreate = !*present_result || framebuffer_resized_;
  if (needs_recreate) {
    framebuffer_resized_ = false;
    if (auto recreated = recreate_swapchain(); !recreated) { return recreated.error(); }
  }

  VkFence submitted = sync.in_flight;
  frame_index_ = (frame_index_ + 1) % static_cast<std::uint32_t>(k_frames);
  frame_open_ = false;
  return submitted;
}

}// namespace vkexec
