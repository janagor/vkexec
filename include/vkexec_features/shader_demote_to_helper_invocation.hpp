#ifndef VKEXEC_FEATURES_SHADER_DEMOTE_TO_HELPER_INVOCATION_HPP
#define VKEXEC_FEATURES_SHADER_DEMOTE_TO_HELPER_INVOCATION_HPP

#include <vkexec_features/feature.hpp>

namespace vkexec::feat {

//! Feature tag for shader demote to helper invocation (core 1.3 / `VK_EXT_shader_demote_to_helper_invocation`).
struct shader_demote_to_helper_invocation
{
};

template<> struct feature_traits<shader_demote_to_helper_invocation>
{
  [[nodiscard]] static constexpr auto name() -> std::string_view { return "shader_demote_to_helper_invocation"; }
  [[nodiscard]] static auto available(context const &ctx) -> bool;
  static auto configure(vulkan_requirements &req) -> void;
};

}// namespace vkexec::feat

#endif// VKEXEC_FEATURES_SHADER_DEMOTE_TO_HELPER_INVOCATION_HPP
