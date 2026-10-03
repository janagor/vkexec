#include <vkexec/context.hpp>
#include <vkexec/execution.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_features/feature.hpp>

#include <stdexec/execution.hpp>

#include <type_traits>

consteval auto cxx20_probe() -> int { return 20; }

static_assert(cxx20_probe() == 20);

auto main() -> int
{
  static_assert(!std::is_copy_constructible_v<vkexec::context>);
  return 0;
}
