#include <vkexec/sampler.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>

#include <vulkan/vulkan_core.h>

namespace vkexec {

auto sampler::create(context &ctx, sampler_create_info info) -> result<sampler>
{
  if (ctx.device() == VK_NULL_HANDLE) { return fail(errc::invalid_argument, "sampler requires a VkDevice"); }

  VkSamplerCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  create_info.magFilter = info.mag_filter;
  create_info.minFilter = info.min_filter;
  create_info.addressModeU = info.address_mode_u;
  create_info.addressModeV = info.address_mode_v;
  create_info.addressModeW = info.address_mode_w;
  create_info.anisotropyEnable = info.anisotropy_enable;
  create_info.maxAnisotropy = info.max_anisotropy;
  create_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  create_info.unnormalizedCoordinates = VK_FALSE;
  create_info.compareEnable = VK_FALSE;
  create_info.compareOp = VK_COMPARE_OP_ALWAYS;
  create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  create_info.mipLodBias = 0.0F;
  create_info.minLod = 0.0F;
  create_info.maxLod = 0.0F;

  VkSampler sampler_handle{ VK_NULL_HANDLE };
  VkResult const create_result = vkCreateSampler(ctx.device(), &create_info, nullptr, &sampler_handle);
  if (create_result != VK_SUCCESS) { return fail(create_result, "vkCreateSampler failed"); }
  return sampler{ &ctx, sampler_handle };
}

sampler::sampler(context *ctx, VkSampler handle) noexcept : ctx_(ctx), sampler_(handle) {}

sampler::~sampler() { destroy(); }

sampler::sampler(sampler &&other) noexcept : ctx_(other.ctx_), sampler_(other.sampler_)
{
  other.ctx_ = nullptr;
  other.sampler_ = VK_NULL_HANDLE;
}

auto sampler::operator=(sampler &&other) noexcept -> sampler &
{
  if (this == &other) { return *this; }
  destroy();
  ctx_ = other.ctx_;
  sampler_ = other.sampler_;
  other.ctx_ = nullptr;
  other.sampler_ = VK_NULL_HANDLE;
  return *this;
}

auto sampler::destroy() noexcept -> void
{
  if (ctx_ != nullptr && ctx_->device() != VK_NULL_HANDLE && sampler_ != VK_NULL_HANDLE) {
    vkDestroySampler(ctx_->device(), sampler_, nullptr);
  }
  ctx_ = nullptr;
  sampler_ = VK_NULL_HANDLE;
}

}// namespace vkexec
