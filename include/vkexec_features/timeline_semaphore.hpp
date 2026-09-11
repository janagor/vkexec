#ifndef VKEXEC_FEATURES_TIMELINE_SEMAPHORE_HPP
#define VKEXEC_FEATURES_TIMELINE_SEMAPHORE_HPP

#include <vkexec_features/feature.hpp>

namespace vkexec::feat {

/**
 * Feature tag for Vulkan timeline semaphores (core 1.2 / `VK_KHR_timeline_semaphore`).
 *
 * @see feature_traits, configure, available
 */
struct timeline_semaphore
{
};

template<> struct feature_traits<timeline_semaphore>
{
  [[nodiscard]] static constexpr auto name() -> std::string_view { return "timeline_semaphore"; }
  [[nodiscard]] static auto available(context const &ctx) -> bool;
  static auto configure(vulkan_requirements &req) -> void;
};

}// namespace vkexec::feat

#endif// VKEXEC_FEATURES_TIMELINE_SEMAPHORE_HPP
