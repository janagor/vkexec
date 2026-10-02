#include <vkexec/context.hpp>

#include <type_traits>

#ifndef VKEXEC_PARENT_VULKAN_HEADERS
#error "vkexec did not use the parent-provided Vulkan::Headers target"
#endif

#ifdef VKEXEC_PARENT_VULKAN_LOADER
#error "Vulkan::Vulkan compile usage requirements leaked through LINK_ONLY"
#endif

consteval auto cxx20_probe() -> int { return 20; }

static_assert(cxx20_probe() == 20);

int main()
{
  static_assert(!std::is_copy_constructible_v<vkexec::context>);
  return 0;
}
