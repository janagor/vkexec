#ifndef VKEXEC_EXTENSIONS_DYNAMIC_RENDERING_EXTENSION_HPP
#define VKEXEC_EXTENSIONS_DYNAMIC_RENDERING_EXTENSION_HPP

#include <vkexec_extensions/extension.hpp>

namespace vkexec::ext {

struct dynamic_rendering {};

template<>
struct extension_traits<dynamic_rendering>
{
  [[nodiscard]] static constexpr auto name() -> std::string_view { return "dynamic_rendering"; }
  [[nodiscard]] static auto available(context const &ctx) -> bool;
  static auto configure(vulkan_requirements &req) -> void;
};

}// namespace vkexec::ext

#endif// VKEXEC_EXTENSIONS_DYNAMIC_RENDERING_EXTENSION_HPP
