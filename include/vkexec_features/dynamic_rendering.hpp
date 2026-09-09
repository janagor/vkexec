#ifndef VKEXEC_FEATURES_DYNAMIC_RENDERING_HPP
#define VKEXEC_FEATURES_DYNAMIC_RENDERING_HPP

#include <vkexec_features/feature.hpp>

namespace vkexec::feat {

struct dynamic_rendering {};

template<>
struct feature_traits<dynamic_rendering>
{
  [[nodiscard]] static constexpr auto name() -> std::string_view { return "dynamic_rendering"; }
  [[nodiscard]] static auto available(context const &ctx) -> bool;
  static auto configure(vulkan_requirements &req) -> void;
};

}// namespace vkexec::feat

#endif// VKEXEC_FEATURES_DYNAMIC_RENDERING_HPP
