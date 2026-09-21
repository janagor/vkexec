#ifndef VKEXEC_SAMPLER_HPP
#define VKEXEC_SAMPLER_HPP

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/sender.hpp>

#include <vulkan/vulkan.h>

namespace vkexec {

//! Creation parameters for `factory::sampler` (defaults to linear clamp-to-edge).
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

class sampler;

namespace factory {

  /**
   * Creates a sampler from `info`.
   *
   * @param ctx Context that owns the device.
   * @param info Filter and addressing parameters.
   */
  [[nodiscard]] auto sampler(::vkexec::context &ctx, sampler_create_info info = {}) -> sender<::vkexec::sampler>;

}// namespace factory

/**
 * RAII `VkSampler` owned by a `context` device.
 *
 * @see factory::sampler, sampler_create_info
 */
class sampler
{
public:
  ~sampler();

  sampler(sampler const &) = delete;
  auto operator=(sampler const &) -> sampler & = delete;

  sampler(sampler &&other) noexcept;
  auto operator=(sampler &&other) noexcept -> sampler &;

  //! Vulkan sampler handle (null after move).
  [[nodiscard]] auto handle() const noexcept -> VkSampler { return sampler_; }

private:
  friend auto factory::sampler(::vkexec::context &ctx, sampler_create_info info) -> sender<::vkexec::sampler>;

  sampler(context *ctx, VkSampler handle) noexcept;
  auto destroy() noexcept -> void;

  context *ctx_{ nullptr };
  VkSampler sampler_{ VK_NULL_HANDLE };
};

}// namespace vkexec

#endif// VKEXEC_SAMPLER_HPP
