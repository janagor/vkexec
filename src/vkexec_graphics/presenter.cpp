#include <vkexec_graphics/presenter.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/presentation_abort_state.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec/sync_wait_outcome.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_graphics/swapchain.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace vkexec {
namespace {

  constexpr std::uint32_t k_color_attachment_index = 0;
  constexpr std::uint32_t k_depth_attachment_index = 1;
  constexpr std::uint32_t k_framebuffer_attachment_count = 2;

  auto pick_depth_format(VkPhysicalDevice phys) -> result<VkFormat>
  {
    static constexpr std::array k_candidates{
      VK_FORMAT_D32_SFLOAT,
      VK_FORMAT_D32_SFLOAT_S8_UINT,
      VK_FORMAT_D24_UNORM_S8_UINT,
    };
    constexpr auto k_depth_stencil_feature =
      static_cast<VkFormatFeatureFlags>(VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    for (VkFormat const format : k_candidates) {
      VkFormatProperties properties{};
      vkGetPhysicalDeviceFormatProperties(phys, format, &properties);
      if ((properties.optimalTilingFeatures & k_depth_stencil_feature) != 0U) { return format; }
    }
    return fail(errc::unsupported, "no supported depth format");
  }

  auto format_has_stencil(VkFormat format) -> bool
  { return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT; }

}// namespace

auto detail::make_presenter_factory::operator()() -> result<::vkexec::owned::presenter>
{
  ::vkexec::owned::presenter created;
  if (auto initialized = created.init(std::move(cfg)); !initialized) { return fail(initialized); }
  return created;
}

auto detail::make_headless_presenter_factory::operator()() -> result<::vkexec::owned::presenter>
{
  auto const surface_exts = vulkan_library::required_headless_surface_instance_extensions();
  cfg.surface_instance_extensions.assign(surface_exts.begin(), surface_exts.end());
  cfg.create_surface = [](VkInstance instance) -> result<VkSurfaceKHR> {
    auto const create_fn =
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT"));
    if (create_fn == nullptr) { return fail(errc::unsupported, "vkCreateHeadlessSurfaceEXT not available"); }

    VkHeadlessSurfaceCreateInfoEXT create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (VkResult const result = create_fn(instance, &create_info, nullptr, &surface); result != VK_SUCCESS) {
      return fail(result, "vkCreateHeadlessSurfaceEXT failed");
    }
    return surface;
  };

  ::vkexec::owned::presenter created;
  if (auto initialized = created.init(std::move(cfg)); !initialized) { return fail(initialized); }
  return created;
}

owned::presenter::presenter(presenter &&other) noexcept
  : cfg_(std::move(other.cfg_)), ctx_(std::move(other.ctx_)), surface_(other.surface_),
    swapchain_(std::move(other.swapchain_)), depth_format_(other.depth_format_),
    framebuffers_(std::move(other.framebuffers_)), depth_(std::move(other.depth_)), render_pass_(other.render_pass_),
    frames_(std::move(other.frames_)), command_pools_(std::move(other.command_pools_)),
    command_buffers_(std::move(other.command_buffers_)), render_finished_(std::move(other.render_finished_)),
    images_in_flight_(std::move(other.images_in_flight_)), frame_index_(other.frame_index_),
    current_image_index_(other.current_image_index_), resize_required_(other.resize_required_),
    suspended_(other.suspended_), frame_open_(other.frame_open_), recording_open_(other.recording_open_),
    acquired_wait_consumed_(other.acquired_wait_consumed_)
{
  other.surface_ = VK_NULL_HANDLE;
  other.render_pass_ = VK_NULL_HANDLE;
  other.frame_open_ = false;
  other.recording_open_ = false;
  other.acquired_wait_consumed_ = false;
}

auto owned::presenter::operator=(presenter &&other) noexcept -> presenter &
{
  if (this == &other) { return *this; }
  this->~presenter();
  new (this) presenter(std::move(other));
  return *this;
}

auto owned::presenter::init(config cfg) -> status
{
  cfg_ = std::move(cfg);
  if (cfg_.width == 0 || cfg_.height == 0) {
    return fail(errc::invalid_argument, "presenter requires a non-zero initial extent");
  }
  if (!cfg_.create_surface) { return fail(errc::invalid_argument, "presenter requires a surface factory"); }
  if (!cfg_.create_depth_attachment) {
    return fail(errc::invalid_argument, "presenter requires a depth attachment factory");
  }

  ctx_ = std::make_unique<::vkexec::context>(context::factory_access{},
    context::instance_only_tag{},
    scheduler_options{
      .validation_layers = cfg_.validation_layers,
      .requirements = cfg_.requirements,
    },
    cfg_.surface_instance_extensions);
  auto created_surface = cfg_.create_surface(ctx_->instance());
  if (!created_surface) { return fail(created_surface); }
  surface_ = expected_take(created_surface);
  if (surface_ == VK_NULL_HANDLE) { return fail(errc::invalid_argument, "surface factory returned a null surface"); }

  if (auto completed = ctx_->complete_for_surface(surface_); !completed) { return fail(completed); }
  if (auto swapchain = create_swapchain(); !swapchain) { return fail(swapchain); }
  if (auto pass = create_render_pass(); !pass) { return fail(pass); }
  if (auto depth = create_depth_resources(); !depth) { return fail(depth); }
  if (auto framebuffers = create_framebuffers(); !framebuffers) { return fail(framebuffers); }
  if (auto frames = create_frame_resources(); !frames) { return fail(frames); }
  return {};
}

owned::presenter::~presenter()
{
  if (ctx_ && ctx_->device() != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(ctx_->device());
    for (auto &sync : frames_) {
      if (sync.in_flight != VK_NULL_HANDLE) { vkDestroyFence(ctx_->device(), sync.in_flight, nullptr); }
      if (sync.image_available != VK_NULL_HANDLE) { vkDestroySemaphore(ctx_->device(), sync.image_available, nullptr); }
    }
    destroy_swapchain_sync();
    for (VkCommandPool pool : command_pools_) {
      if (pool != VK_NULL_HANDLE) { vkDestroyCommandPool(ctx_->device(), pool, nullptr); }
    }
    cleanup_swapchain();
    if (render_pass_ != VK_NULL_HANDLE) { vkDestroyRenderPass(ctx_->device(), render_pass_, nullptr); }
  }
  if (ctx_ && ctx_->instance() != VK_NULL_HANDLE && surface_ != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(ctx_->instance(), surface_, nullptr);
    surface_ = VK_NULL_HANDLE;
  }
  ctx_.reset();
}

auto owned::presenter::wait_idle() -> void
{
  if (ctx_ && ctx_->device() != VK_NULL_HANDLE) { vkDeviceWaitIdle(ctx_->device()); }
}

auto owned::presenter::create_swapchain() -> status
{
  if (!swapchain_.has_value()) {
    auto outcome = try_sync_wait(factory::make_swapchain(*ctx_,
      swapchain_create_info{
        .surface = surface_,
        .width = cfg_.width,
        .height = cfg_.height,
      }));
    if (outcome.error.has_value()) { return fail(std::move(*outcome.error)); }
    if (outcome.stopped || !outcome.values.has_value()) { return fail(errc::cancelled, "swapchain create stopped"); }
    swapchain_.emplace(detail::take_sync_value(std::move(*outcome.values)));
  } else if (auto recreated = swapchain_->recreate(cfg_.width, cfg_.height); !recreated) {
    return fail(recreated);
  }
  return create_swapchain_sync();
}

auto owned::presenter::create_render_pass() -> status
{
  if (depth_format_ == VK_FORMAT_UNDEFINED) {
    auto format = pick_depth_format(ctx_->physical_device());
    if (!format) { return fail(format); }
    depth_format_ = expected_take(format);
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

  auto const attachment_stages = static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
                                 | static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT)
                                 | static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);
  auto const attachment_access = static_cast<VkAccessFlags>(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
                                 | static_cast<VkAccessFlags>(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

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
    return fail(result, "vkCreateRenderPass failed");
  }
  return {};
}

auto owned::presenter::create_depth_resources() -> status
{
  VKEXEC_TRY_ASSIGN(created, cfg_.create_depth_attachment(*ctx_, extent(), depth_format_));
  if (created.image() == VK_NULL_HANDLE || created.view() == VK_NULL_HANDLE || created.format() != depth_format_
      || created.extent().width != extent().width || created.extent().height != extent().height) {
    return fail(errc::invalid_argument, "depth attachment does not match the presenter extent or format");
  }
  depth_ = std::move(created);
  return {};
}

auto owned::presenter::destroy_depth_resources() noexcept -> void { depth_ = {}; }

auto owned::presenter::create_framebuffers() -> status
{
  if (!swapchain_) { return fail(errc::invalid_argument, "create_framebuffers requires a swapchain"); }
  swapchain const &active_swapchain = *swapchain_;
  auto const views = active_swapchain.image_views();
  framebuffers_.resize(views.size());
  for (std::size_t index = 0; index < views.size(); ++index) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    std::array const attachments{ views[index], depth_.view() };
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
      return fail(result, "vkCreateFramebuffer failed");
    }
  }
  return {};
}

auto owned::presenter::create_frame_resources() -> status
{
  frames_.resize(static_cast<std::size_t>(k_frames));
  command_pools_.resize(static_cast<std::size_t>(k_frames));
  command_buffers_.resize(static_cast<std::size_t>(k_frames));
  for (int index = 0; index < k_frames; ++index) {
    auto const frame_index = static_cast<std::size_t>(index);
    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    if (VkResult const result =
          vkCreateSemaphore(ctx_->device(), &semaphore_info, nullptr, &frames_.at(frame_index).image_available);
      result != VK_SUCCESS) {
      return fail(result, "vkCreateSemaphore failed");
    }

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (VkResult const result = vkCreateFence(ctx_->device(), &fence_info, nullptr, &frames_.at(frame_index).in_flight);
      result != VK_SUCCESS) {
      return fail(result, "vkCreateFence failed");
    }

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = ctx_->graphics_queue_family();
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (VkResult const result =
          vkCreateCommandPool(ctx_->device(), &pool_info, nullptr, &command_pools_.at(frame_index));
      result != VK_SUCCESS) {
      return fail(result, "vkCreateCommandPool failed (graphics)");
    }
    VkCommandBufferAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.commandPool = command_pools_.at(frame_index);
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandBufferCount = 1;
    if (VkResult const result =
          vkAllocateCommandBuffers(ctx_->device(), &allocate_info, &command_buffers_.at(frame_index));
      result != VK_SUCCESS) {
      return fail(result, "vkAllocateCommandBuffers failed (graphics)");
    }
  }
  return {};
}

auto owned::presenter::create_swapchain_sync() -> status
{
  if (!swapchain_) { return fail(errc::invalid_argument, "create_swapchain_sync requires a swapchain"); }
  swapchain const &active_swapchain = *swapchain_;
  destroy_swapchain_sync();

  auto const image_count = active_swapchain.images().size();
  render_finished_.resize(image_count);
  images_in_flight_.assign(image_count, VK_NULL_HANDLE);

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  for (VkSemaphore &render_finished : render_finished_) {
    if (VkResult const result = vkCreateSemaphore(ctx_->device(), &semaphore_info, nullptr, &render_finished);
      result != VK_SUCCESS) {
      return fail(result, "vkCreateSemaphore failed (render_finished)");
    }
  }
  return {};
}

auto owned::presenter::destroy_swapchain_sync() noexcept -> void
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

auto owned::presenter::cleanup_swapchain() -> void
{
  destroy_swapchain_sync();
  for (VkFramebuffer framebuffer : framebuffers_) {
    if (framebuffer != VK_NULL_HANDLE) { vkDestroyFramebuffer(ctx_->device(), framebuffer, nullptr); }
  }
  framebuffers_.clear();
  destroy_depth_resources();
  swapchain_.reset();
}

auto owned::presenter::resize(std::uint32_t width, std::uint32_t height) -> status
{
  if (frame_open_) { return fail(errc::invalid_argument, "resize called while a frame is open"); }
  if (width == 0 || height == 0) {
    suspended_ = true;
    resize_required_ = true;
    return {};
  }
  if (!resize_required_ && !suspended_ && width == cfg_.width && height == cfg_.height) { return {}; }
  return recreate_swapchain(width, height);
}

auto owned::presenter::recreate_swapchain(std::uint32_t width, std::uint32_t height) -> status
{
  cfg_.width = width;
  cfg_.height = height;

  // Idle before destroying framebuffers/views that may still be referenced by in-flight frames.
  if (VkResult const idle = vkDeviceWaitIdle(ctx_->device()); idle != VK_SUCCESS) {
    return fail(idle, "vkDeviceWaitIdle failed before swapchain recreation");
  }

  for (VkFramebuffer framebuffer : framebuffers_) {
    if (framebuffer != VK_NULL_HANDLE) { vkDestroyFramebuffer(ctx_->device(), framebuffer, nullptr); }
  }
  framebuffers_.clear();
  destroy_depth_resources();

  if (auto swapchain = create_swapchain(); !swapchain) { return fail(swapchain); }
  if (auto depth = create_depth_resources(); !depth) { return fail(depth); }
  if (auto framebuffers = create_framebuffers(); !framebuffers) { return fail(framebuffers); }
  suspended_ = false;
  resize_required_ = false;
  return {};
}

auto owned::presenter::acquire_frame() -> result<std::optional<acquired_frame>>
{
  if (frame_open_) { return fail(errc::invalid_argument, "acquire_frame called while a frame is already open"); }
  if (suspended_ || resize_required_) { return std::optional<acquired_frame>{}; }
  if (!swapchain_) { return fail(errc::invalid_argument, "acquire_frame requires a swapchain"); }
  swapchain &active_swapchain = *swapchain_;

  auto &sync = frames_.at(frame_index_);
  if (VkResult const wait_result = vkWaitForFences(ctx_->device(), 1, &sync.in_flight, VK_TRUE, UINT64_MAX);
    wait_result != VK_SUCCESS) {
    return fail(wait_result, "vkWaitForFences failed");
  }

  auto acquired = active_swapchain.acquire_next_image(sync.image_available);
  if (!acquired) { return fail(acquired); }
  if (!acquired->has_value()) {
    resize_required_ = true;
    return std::optional<acquired_frame>{};
  }
  std::uint32_t const image_index = **acquired;

  // Wait if a previous frame is still using this swapchain image.
  if (images_in_flight_.at(image_index) != VK_NULL_HANDLE) {
    if (VkResult const wait_result =
          vkWaitForFences(ctx_->device(), 1, &images_in_flight_.at(image_index), VK_TRUE, UINT64_MAX);
      wait_result != VK_SUCCESS) {
      return fail(wait_result, "vkWaitForFences failed (swapchain image in flight)");
    }
  }
  images_in_flight_.at(image_index) = sync.in_flight;

  current_image_index_ = image_index;
  frame_open_ = true;
  recording_open_ = false;
  acquired_wait_consumed_ = false;
  return acquired_frame{ .image = active_swapchain.image_at(image_index),
    .image_view = active_swapchain.image_view_at(image_index),
    .framebuffer = framebuffers_.at(image_index),
    .extent = extent(),
    .image_index = image_index };
}

auto owned::presenter::begin_frame() -> result<std::optional<frame>>
{
  VKEXEC_TRY_ASSIGN(acquired, acquire_frame());
  if (!acquired) { return std::optional<frame>{}; }
  VkFence fence = frames_.at(frame_index_).in_flight;
  if (VkResult const reset_result = vkResetFences(ctx_->device(), 1, &fence); reset_result != VK_SUCCESS) {
    if (auto abandoned = abandon_frame(*acquired); !abandoned) { return fail(abandoned); }
    return fail(reset_result, "vkResetFences failed");
  }
  VkCommandBuffer cmd = command_buffers_.at(frame_index_);
  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (VkResult const begin_result = vkBeginCommandBuffer(cmd, &begin_info); begin_result != VK_SUCCESS) {
    if (auto abandoned = abandon_frame(*acquired); !abandoned) { return fail(abandoned); }
    return fail(begin_result, "vkBeginCommandBuffer failed");
  }
  recording_open_ = true;
  return frame{ .command_buffer = cmd,
    .framebuffer = acquired->framebuffer,
    .extent = acquired->extent,
    .image_index = acquired->image_index };
}

auto owned::presenter::end_frame(frame const &drawn, present_options options) -> result<VkFence>
{ return end_frame(drawn, frame_submit_options{}, options); }

auto owned::presenter::end_frame(frame const &drawn, frame_submit_options submit_options, present_options options)
  -> result<VkFence>
{
  if (recording_open_) { VKEXEC_TRY(finish_frame_recording(drawn)); }
  VKEXEC_TRY_ASSIGN(sync, submission_sync(drawn));
  std::array<VkCommandBuffer, 1> const commands{ drawn.command_buffer };
  std::vector<semaphore_submit> waits;
  waits.reserve(1 + submit_options.waits.size());
  waits.push_back(sync.image_available_wait);
  waits.insert(waits.end(), submit_options.waits.begin(), submit_options.waits.end());
  std::array<semaphore_submit, 1> const signals{ sync.render_finished_signal };
  VKEXEC_TRY(ctx_->submit(queue_submit{ .command_buffers = commands,
    .waits = waits,
    .signals = signals,
    .fence = sync.fence,
    .queue = ctx_->graphics_queue() }));

  return present_submitted(drawn, options);
}

auto owned::presenter::finish_frame_recording(frame const &drawn) -> status
{
  VKEXEC_TRY(validate_current_frame(drawn));
  if (!recording_open_) { return fail(errc::invalid_argument, "frame command buffer is not recording"); }
  if (VkResult const result = vkEndCommandBuffer(drawn.command_buffer); result != VK_SUCCESS) {
    return fail(result, "vkEndCommandBuffer failed");
  }
  recording_open_ = false;
  return {};
}

auto owned::presenter::validate_current_frame(frame const &drawn) const -> status
{
  if (drawn.command_buffer != command_buffers_.at(frame_index_)) {
    return fail(errc::invalid_argument, "operation requires the current open frame");
  }
  return validate_current_frame(
    acquired_frame{ .framebuffer = drawn.framebuffer, .extent = drawn.extent, .image_index = drawn.image_index });
}

auto owned::presenter::validate_current_frame(acquired_frame const &drawn) const -> status
{
  if (!frame_open_ || !swapchain_ || drawn.image_index != current_image_index_
      || drawn.framebuffer != framebuffers_.at(current_image_index_) || drawn.extent.width != extent().width
      || drawn.extent.height != extent().height
      || (drawn.image != VK_NULL_HANDLE && drawn.image != swapchain_->image_at(current_image_index_))
      || (drawn.image_view != VK_NULL_HANDLE && drawn.image_view != swapchain_->image_view_at(current_image_index_))) {
    return fail(errc::invalid_argument, "operation requires the current open frame");
  }
  return {};
}

auto owned::presenter::submission_sync(frame const &drawn) const -> result<frame_submit_sync>
{
  VKEXEC_TRY(validate_current_frame(drawn));
  return submission_sync(
    acquired_frame{ .framebuffer = drawn.framebuffer, .extent = drawn.extent, .image_index = drawn.image_index });
}

auto owned::presenter::submission_sync(acquired_frame const &drawn) const -> result<frame_submit_sync>
{
  VKEXEC_TRY(validate_current_frame(drawn));
  frame_sync const &sync = frames_.at(frame_index_);
  return frame_submit_sync{
    .image_available_wait =
      semaphore_submit{ .semaphore = sync.image_available, .stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT },
    .render_finished_signal = semaphore_submit{ .semaphore = render_finished_.at(current_image_index_) },
    .fence = sync.in_flight,
  };
}

auto owned::presenter::present_submitted(frame const &drawn, present_options options) -> result<VkFence>
{
  VKEXEC_TRY(validate_current_frame(drawn));
  if (recording_open_) { return fail(errc::invalid_argument, "frame command buffer is still recording"); }
  return present_submitted(
    acquired_frame{ .framebuffer = drawn.framebuffer, .extent = drawn.extent, .image_index = drawn.image_index },
    options);
}

auto owned::presenter::present_submitted(acquired_frame const &drawn, present_options options) -> result<VkFence>
{
  VKEXEC_TRY(validate_current_frame(drawn));
  if (recording_open_) { return fail(errc::invalid_argument, "frame command buffer is still recording"); }
  VKEXEC_TRY_ASSIGN(sync, submission_sync(drawn));
  if (!swapchain_) { return fail(errc::invalid_argument, "present_submitted requires a swapchain"); }
  swapchain &active_swapchain = *swapchain_;

  std::array<VkSemaphore, 1> const wait_semaphores{ sync.render_finished_signal.semaphore };
  auto present_result = active_swapchain.present(current_image_index_, wait_semaphores, options);
  if (!present_result) { return fail(present_result); }
  if (!*present_result) { resize_required_ = true; }

  // Caller may enqueue_borrowed_fence_wait on this fence; presenter retains ownership.
  VkFence submitted = sync.fence;
  frame_index_ = (frame_index_ + 1) % static_cast<std::uint32_t>(k_frames);
  frame_open_ = false;
  recording_open_ = false;
  acquired_wait_consumed_ = false;
  return submitted;
}

auto owned::presenter::abandon_frame(acquired_frame const &drawn, presentation_abort_state state) -> status
{
  VKEXEC_TRY(validate_current_frame(drawn));
  if (recording_open_) { return fail(errc::invalid_argument, "cannot abandon a recording frame"); }
  if (VkResult const idle = vkDeviceWaitIdle(ctx_->device()); idle != VK_SUCCESS) {
    return fail(idle, "vkDeviceWaitIdle failed while abandoning frame");
  }
  if (state == presentation_abort_state::acquired_only && !acquired_wait_consumed_) {
    VKEXEC_TRY(consume_acquired_image_wait());
    acquired_wait_consumed_ = true;
  }
  if (state == presentation_abort_state::final_submit_completed) {
    VKEXEC_TRY(replace_abandoned_frame_sync());
    frame_open_ = false;
    acquired_wait_consumed_ = false;
    cleanup_swapchain();
    resize_required_ = true;
    return {};
  }
  if (!swapchain_.has_value()) { return fail(errc::invalid_argument, "abandon_frame requires a swapchain"); }
  auto presented = swapchain_.value().present(current_image_index_, {});
  if (VkResult const idle = vkDeviceWaitIdle(ctx_->device()); idle != VK_SUCCESS) {
    return fail(idle, "vkDeviceWaitIdle failed after abandoned frame presentation");
  }
  VKEXEC_TRY(replace_abandoned_frame_sync());
  frame_open_ = false;
  acquired_wait_consumed_ = false;
  if (!presented || !*presented) {
    cleanup_swapchain();
    resize_required_ = true;
    if (!presented) { return fail(presented); }
  }
  return {};
}

auto owned::presenter::consume_acquired_image_wait() -> status
{
  VkCommandBuffer cmd = command_buffers_.at(frame_index_);
  if (VkResult const reset = vkResetCommandBuffer(cmd, 0); reset != VK_SUCCESS) {
    return fail(reset, "vkResetCommandBuffer failed while abandoning frame");
  }
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (VkResult const begun = vkBeginCommandBuffer(cmd, &begin_info); begun != VK_SUCCESS) {
    return fail(begun, "vkBeginCommandBuffer failed while abandoning frame");
  }
  std::array<VkClearValue, k_framebuffer_attachment_count> const clears{
    VkClearValue{ .color = VkClearColorValue{ .float32 = { 0.0F, 0.0F, 0.0F, 1.0F } } },
    VkClearValue{ .depthStencil = VkClearDepthStencilValue{ .depth = 1.0F, .stencil = 0 } },
  };
  VkRenderPassBeginInfo render_begin{};
  render_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  render_begin.renderPass = render_pass_;
  render_begin.framebuffer = framebuffers_.at(current_image_index_);
  render_begin.renderArea.extent = extent();
  render_begin.clearValueCount = static_cast<std::uint32_t>(clears.size());
  render_begin.pClearValues = clears.data();
  vkCmdBeginRenderPass(cmd, &render_begin, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdEndRenderPass(cmd);
  if (VkResult const ended = vkEndCommandBuffer(cmd); ended != VK_SUCCESS) {
    return fail(ended, "vkEndCommandBuffer failed while abandoning frame");
  }
  VkFenceCreateInfo consume_info{};
  consume_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence consumed{ VK_NULL_HANDLE };
  if (VkResult const created = vkCreateFence(ctx_->device(), &consume_info, nullptr, &consumed);
    created != VK_SUCCESS) {
    return fail(created, "vkCreateFence failed while abandoning frame");
  }
  std::array<VkCommandBuffer, 1> const commands{ cmd };
  std::array<semaphore_submit, 1> const waits{ semaphore_submit{
    .semaphore = frames_.at(frame_index_).image_available } };
  auto submitted = ctx_->submit(
    queue_submit{ .command_buffers = commands, .waits = waits, .fence = consumed, .queue = ctx_->graphics_queue() });
  if (!submitted) {
    vkDestroyFence(ctx_->device(), consumed, nullptr);
    return fail(submitted);
  }
  VkResult const waited = vkWaitForFences(ctx_->device(), 1, &consumed, VK_TRUE, UINT64_MAX);
  vkDestroyFence(ctx_->device(), consumed, nullptr);
  if (waited != VK_SUCCESS) { return fail(waited, "vkWaitForFences failed while abandoning frame"); }
  return {};
}

auto owned::presenter::replace_abandoned_frame_sync() -> status
{
  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VkSemaphore image_available{ VK_NULL_HANDLE };
  VkSemaphore render_finished{ VK_NULL_HANDLE };
  VkFence in_flight{ VK_NULL_HANDLE };
  auto cleanup = [&]() -> void {
    if (image_available != VK_NULL_HANDLE) { vkDestroySemaphore(ctx_->device(), image_available, nullptr); }
    if (render_finished != VK_NULL_HANDLE) { vkDestroySemaphore(ctx_->device(), render_finished, nullptr); }
    if (in_flight != VK_NULL_HANDLE) { vkDestroyFence(ctx_->device(), in_flight, nullptr); }
  };
  if (VkResult const created = vkCreateSemaphore(ctx_->device(), &semaphore_info, nullptr, &image_available);
    created != VK_SUCCESS) {
    return fail(created, "vkCreateSemaphore failed while abandoning frame");
  }
  if (VkResult const created = vkCreateSemaphore(ctx_->device(), &semaphore_info, nullptr, &render_finished);
    created != VK_SUCCESS) {
    cleanup();
    return fail(created, "vkCreateSemaphore failed while abandoning frame");
  }
  if (VkResult const created = vkCreateFence(ctx_->device(), &fence_info, nullptr, &in_flight); created != VK_SUCCESS) {
    cleanup();
    return fail(created, "vkCreateFence failed while abandoning frame");
  }

  auto &sync = frames_.at(frame_index_);
  for (VkFence &image_fence : images_in_flight_) {
    if (image_fence == sync.in_flight) { image_fence = VK_NULL_HANDLE; }
  }
  vkDestroySemaphore(ctx_->device(), sync.image_available, nullptr);
  vkDestroySemaphore(ctx_->device(), render_finished_.at(current_image_index_), nullptr);
  vkDestroyFence(ctx_->device(), sync.in_flight, nullptr);
  sync.image_available = image_available;
  sync.in_flight = in_flight;
  render_finished_.at(current_image_index_) = render_finished;
  return {};
}

}// namespace vkexec
