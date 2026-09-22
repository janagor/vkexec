#ifndef VKEXEC_SAMPLER_HPP
#define VKEXEC_SAMPLER_HPP

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/sender.hpp>

#include <vulkan/vulkan.h>

namespace vkexec {

//! Creation parameters for `factory::make_sampler` (defaults to linear clamp-to-edge).
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

namespace owned {
  class sampler;
}// namespace owned

namespace factory {

  struct make_sampler_t
  {

    /**
     * Creates a sampler from `info`.
     *
     * @param ctx Context that owns the device.
     * @param info Filter and addressing parameters.
     */
    [[nodiscard]] auto operator()(context &ctx, sampler_create_info info = {}) const -> sender<owned::sampler>;
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_sampler_t make_sampler{};

}// namespace factory

/**
 * RAII `VkSampler` owned by a `context` device.
 *
 * @see factory::make_sampler, sampler_create_info
 */
namespace owned {

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
    friend struct factory::make_sampler_t;

    sampler(context *ctx, VkSampler handle) noexcept;
    auto destroy() noexcept -> void;

    context *ctx_{ nullptr };
    VkSampler sampler_{ VK_NULL_HANDLE };
  };

}// namespace owned

}// namespace vkexec

#endif// VKEXEC_SAMPLER_HPP
