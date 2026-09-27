#include <vkexec/context.hpp>
#include <vkexec/execution.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_features/feature.hpp>

#include <stdexec/execution.hpp>

#include <type_traits>

int main()
{
  static_assert(!std::is_copy_constructible_v<vkexec::context>);
  return 0;
}
