#include <vkexec_vma/offscreen_target.hpp>

#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/image_view.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_vma/allocator.hpp>
#include <vkexec_vma/image.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace vkexec::vma {

struct offscreen_target::state
{
  state() = default;
  state(state const &) = delete;
  auto operator=(state const &) -> state & = delete;
  state(state &&) = delete;
  auto operator=(state &&) -> state & = delete;
  context *ctx{ nullptr };
  VkExtent2D extent{};
  std::vector<image> colors;
  std::vector<owned::image_view> color_views;
  std::optional<image> depth;
  std::optional<owned::image_view> depth_view;
  VkRenderPass pass{ VK_NULL_HANDLE };
  VkFramebuffer framebuffer{ VK_NULL_HANDLE };

  ~state()
  {
    if (framebuffer != VK_NULL_HANDLE) { vkDestroyFramebuffer(ctx->device(), framebuffer, nullptr); }
    if (pass != VK_NULL_HANDLE) { vkDestroyRenderPass(ctx->device(), pass, nullptr); }
  }
};

offscreen_target::offscreen_target(std::unique_ptr<state> owned) noexcept : state_(std::move(owned)) {}
offscreen_target::offscreen_target(offscreen_target &&) noexcept = default;
auto offscreen_target::operator=(offscreen_target &&) noexcept -> offscreen_target & = default;
offscreen_target::~offscreen_target() = default;

auto offscreen_target::create(context &ctx, allocator &alloc, offscreen_target_config const &config)
  -> result<offscreen_target>
{
  if (config.extent.width == 0 || config.extent.height == 0
      || (config.color_formats.empty() && config.depth_format == VK_FORMAT_UNDEFINED)) {
    return fail(errc::invalid_argument, "offscreen target needs an extent and attachment format");
  }
  auto target = std::make_unique<state>();
  target->ctx = &ctx;
  target->extent = config.extent;
  std::vector<VkAttachmentDescription> attachments;
  std::vector<VkAttachmentReference> color_refs;
  std::vector<VkImageView> framebuffer_views;
  attachments.reserve(config.color_formats.size() + 1);
  color_refs.reserve(config.color_formats.size());
  framebuffer_views.reserve(config.color_formats.size() + 1);

  for (VkFormat const format : config.color_formats) {
    auto allocated = try_sync_wait_value(allocate_image(alloc,
      ::vkexec::image_create_info{
        .extent = VkExtent3D{ .width = config.extent.width, .height = config.extent.height, .depth = 1 },
        .format = format,
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT }));
    if (!allocated) { return fail(allocated); }
    target->colors.push_back(std::move(*allocated));
    auto view = try_sync_wait_value(factory::make_image_view(ctx, target->colors.back()));
    if (!view) { return fail(view); }
    target->color_views.push_back(std::move(*view));
    framebuffer_views.push_back(target->color_views.back().handle());
    color_refs.push_back(VkAttachmentReference{
      .attachment = static_cast<std::uint32_t>(color_refs.size()),
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    });
    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
    VkAttachmentDescription attachment{};
    attachment.format = format;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments.push_back(attachment);
  }

  VkAttachmentReference depth_ref{};
  if (config.depth_format != VK_FORMAT_UNDEFINED) {
    auto allocated = try_sync_wait_value(allocate_image(alloc,
      ::vkexec::image_create_info{
        .extent = VkExtent3D{ .width = config.extent.width, .height = config.extent.height, .depth = 1 },
        .format = config.depth_format,
        .usage = static_cast<VkImageUsageFlags>(VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
                 | (config.sample_depth ? static_cast<VkImageUsageFlags>(VK_IMAGE_USAGE_SAMPLED_BIT) : 0U) }));
    if (!allocated) { return fail(allocated); }
    target->depth.emplace(std::move(*allocated));
    auto view = try_sync_wait_value(factory::make_image_view(ctx, *target->depth));
    if (!view) { return fail(view); }
    target->depth_view.emplace(std::move(*view));
    framebuffer_views.push_back(target->depth_view->handle());
    depth_ref = VkAttachmentReference{
      .attachment = static_cast<std::uint32_t>(attachments.size()),
      .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
    VkAttachmentDescription attachment{};
    attachment.format = config.depth_format;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = config.sample_depth ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments.push_back(attachment);
  }

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = static_cast<std::uint32_t>(color_refs.size());
  subpass.pColorAttachments = color_refs.data();
  subpass.pDepthStencilAttachment = target->depth.has_value() ? &depth_ref : nullptr;
  VkRenderPassCreateInfo pass_info{};
  pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  pass_info.attachmentCount = static_cast<std::uint32_t>(attachments.size());
  pass_info.pAttachments = attachments.data();
  pass_info.subpassCount = 1;
  pass_info.pSubpasses = &subpass;
  if (VkResult const created = vkCreateRenderPass(ctx.device(), &pass_info, nullptr, &target->pass);
    created != VK_SUCCESS) {
    return fail(created, "vkCreateRenderPass failed for offscreen target");
  }
  VkFramebufferCreateInfo framebuffer_info{};
  framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebuffer_info.renderPass = target->pass;
  framebuffer_info.attachmentCount = static_cast<std::uint32_t>(framebuffer_views.size());
  framebuffer_info.pAttachments = framebuffer_views.data();
  framebuffer_info.width = config.extent.width;
  framebuffer_info.height = config.extent.height;
  framebuffer_info.layers = 1;
  if (VkResult const created = vkCreateFramebuffer(ctx.device(), &framebuffer_info, nullptr, &target->framebuffer);
    created != VK_SUCCESS) {
    return fail(created, "vkCreateFramebuffer failed for offscreen target");
  }
  return offscreen_target{ std::move(target) };
}

auto offscreen_target::render_pass() const noexcept -> VkRenderPass { return state_->pass; }
auto offscreen_target::framebuffer() const noexcept -> VkFramebuffer { return state_->framebuffer; }
auto offscreen_target::extent() const noexcept -> VkExtent2D { return state_->extent; }
auto offscreen_target::color_image(std::size_t index) const -> VkImage { return state_->colors.at(index).handle(); }
auto offscreen_target::color_view(std::size_t index) const -> VkImageView
{ return state_->color_views.at(index).handle(); }
auto offscreen_target::depth_image() const noexcept -> VkImage
{ return state_->depth.has_value() ? state_->depth->handle() : VK_NULL_HANDLE; }
auto offscreen_target::depth_view() const noexcept -> VkImageView
{ return state_->depth_view.has_value() ? state_->depth_view->handle() : VK_NULL_HANDLE; }
auto offscreen_target::clear_values() const -> std::vector<VkClearValue>
{
  std::vector<VkClearValue> clears(state_->colors.size());
  for (VkClearValue &clear : clears) { clear.color = VkClearColorValue{ .float32 = { 0.0F, 0.0F, 0.0F, 1.0F } }; }
  if (state_->depth.has_value()) {
    clears.push_back(VkClearValue{ .depthStencil = VkClearDepthStencilValue{ .depth = 1.0F, .stencil = 0 } });
  }
  return clears;
}

}// namespace vkexec::vma
