#ifndef VKEXEC_SAMPLER_HPP
#define VKEXEC_SAMPLER_HPP

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>

#include <vulkan/vulkan.h>

namespace vkexec {

struct sampler_create_info
{
  VkFilter mag_filter{ VK_FILTER_LINEAR };
  VkFilter min_filter{ VK_FILTER_LINEAR };
  VkSamplerAddressMode address_mode_u{ VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE };
  VkSamplerAddressMode address_mode_v{ VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE };
  VkSamplerAddressMode address_mode_w{ VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE };
  VkBool32 anisotropy_enable{ VK_FALSE };
  float max_anisotropy{ 1.0F };
};

/// RAII `VkSampler`.
class sampler
{
public:
  [[nodiscard]] static auto create(context &ctx, sampler_create_info info = {}) -> sampler;

  ~sampler();

  sampler(sampler const &) = delete;
  auto operator=(sampler const &) -> sampler & = delete;

  sampler(sampler &&other) noexcept;
  auto operator=(sampler &&other) noexcept -> sampler &;

  [[nodiscard]] auto handle() const noexcept -> VkSampler { return sampler_; }

private:
  sampler(context *ctx, VkSampler handle) noexcept;
  auto destroy() noexcept -> void;

  context *ctx_{ nullptr };
  VkSampler sampler_{ VK_NULL_HANDLE };
};

}// namespace vkexec

#endif// VKEXEC_SAMPLER_HPP
