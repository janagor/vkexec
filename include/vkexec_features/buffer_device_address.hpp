#ifndef VKEXEC_FEATURES_BUFFER_DEVICE_ADDRESS_HPP
#define VKEXEC_FEATURES_BUFFER_DEVICE_ADDRESS_HPP

#include <vkexec_features/feature.hpp>

namespace vkexec::feat {

/**
 * Feature tag for buffer device address (core 1.2 / `VK_KHR_buffer_device_address`).
 *
 * @see feature_traits, configure, available, gpu_buffer
 */
struct buffer_device_address
{
};

template<> struct feature_traits<buffer_device_address>
{
  [[nodiscard]] static constexpr auto name() -> std::string_view { return "buffer_device_address"; }
  [[nodiscard]] static auto available(context const &ctx) -> bool;
  static auto configure(vulkan_requirements &req) -> void;
};

}// namespace vkexec::feat

#endif// VKEXEC_FEATURES_BUFFER_DEVICE_ADDRESS_HPP
