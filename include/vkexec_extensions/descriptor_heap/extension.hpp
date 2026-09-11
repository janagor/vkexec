#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_EXTENSION_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_EXTENSION_HPP

#include <vkexec_extensions/extension.hpp>

namespace vkexec::ext {

/**
 * Extension tag for `VK_EXT_descriptor_heap` bindless resource heaps.
 *
 * @see extension_traits, configure, available
 */
struct descriptor_heap
{
};

template<> struct extension_traits<descriptor_heap>
{
  [[nodiscard]] static constexpr auto name() -> std::string_view { return "descriptor_heap"; }
  [[nodiscard]] static auto available(context const &ctx) -> bool;
  static auto configure(vulkan_requirements &req) -> void;
};

}// namespace vkexec::ext

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_EXTENSION_HPP
