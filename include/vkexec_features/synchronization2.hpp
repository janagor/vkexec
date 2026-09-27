#ifndef VKEXEC_FEATURES_SYNCHRONIZATION2_HPP
#define VKEXEC_FEATURES_SYNCHRONIZATION2_HPP

#include <vkexec_features/feature.hpp>

namespace vkexec::feat {

//! Feature tag for synchronization2 (core 1.3 / `VK_KHR_synchronization2`).
struct synchronization2
{
};

template<> struct feature_traits<synchronization2>
{
  [[nodiscard]] static constexpr auto name() -> std::string_view { return "synchronization2"; }
  [[nodiscard]] static auto available(context const &ctx) -> bool;
  static auto configure(vulkan_requirements &req) -> void;
};

}// namespace vkexec::feat

#endif// VKEXEC_FEATURES_SYNCHRONIZATION2_HPP
