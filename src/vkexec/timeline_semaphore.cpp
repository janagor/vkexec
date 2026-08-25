#include <vkexec/timeline_semaphore.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error_helpers.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <limits>

namespace vkexec {

auto timeline_semaphore::create(context &ctx, std::uint64_t initial_value) -> result<timeline_semaphore>
{
  if (ctx.device() == VK_NULL_HANDLE) {
    return std::unexpected(make_error(errc::invalid_argument, "timeline_semaphore requires a VkDevice"));
  }

  VkSemaphoreTypeCreateInfo type_info{};
  type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  type_info.initialValue = initial_value;

  VkSemaphoreCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  info.pNext = &type_info;

  VkSemaphore semaphore{ VK_NULL_HANDLE };
  VkResult const create_result = vkCreateSemaphore(ctx.device(), &info, nullptr, &semaphore);
  if (create_result != VK_SUCCESS) {
    return std::unexpected(make_vk_error(create_result, "vkCreateSemaphore (timeline) failed"));
  }
  return timeline_semaphore{ &ctx, semaphore };
}

timeline_semaphore::timeline_semaphore(context *ctx, VkSemaphore semaphore) noexcept
  : ctx_(ctx), semaphore_(semaphore)
{}

timeline_semaphore::~timeline_semaphore() { destroy(); }

timeline_semaphore::timeline_semaphore(timeline_semaphore &&other) noexcept
  : ctx_(other.ctx_), semaphore_(other.semaphore_)
{
  other.ctx_ = nullptr;
  other.semaphore_ = VK_NULL_HANDLE;
}

auto timeline_semaphore::operator=(timeline_semaphore &&other) noexcept -> timeline_semaphore &
{
  if (this == &other) { return *this; }
  destroy();
  ctx_ = other.ctx_;
  semaphore_ = other.semaphore_;
  other.ctx_ = nullptr;
  other.semaphore_ = VK_NULL_HANDLE;
  return *this;
}

auto timeline_semaphore::wait(std::uint64_t value) const -> status
{
  if (value == 0 || semaphore_ == VK_NULL_HANDLE) { return {}; }
  if (ctx_ == nullptr) {
    return make_error(errc::invalid_argument, "timeline_semaphore::wait requires a live context");
  }

  std::array<VkSemaphore, 1> const semaphores{ semaphore_ };
  std::array<std::uint64_t, 1> const values{ value };

  VkSemaphoreWaitInfo wait_info{};
  wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  wait_info.semaphoreCount = 1;
  wait_info.pSemaphores = semaphores.data();
  wait_info.pValues = values.data();

  VkResult const wait_result =
    vkWaitSemaphores(ctx_->device(), &wait_info, std::numeric_limits<std::uint64_t>::max());
  if (wait_result != VK_SUCCESS) { return make_vk_error(wait_result, "vkWaitSemaphores failed"); }
  return {};
}

auto timeline_semaphore::destroy() noexcept -> void
{
  if (ctx_ != nullptr && ctx_->device() != VK_NULL_HANDLE && semaphore_ != VK_NULL_HANDLE) {
    vkDestroySemaphore(ctx_->device(), semaphore_, nullptr);
  }
  ctx_ = nullptr;
  semaphore_ = VK_NULL_HANDLE;
}

}// namespace vkexec
