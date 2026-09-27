#include <vkexec/context.hpp>

#include <type_traits>

consteval auto cxx20_probe() -> int { return 20; }

static_assert(cxx20_probe() == 20);

int main()
{
  static_assert(!std::is_copy_constructible_v<vkexec::context>);
  return 0;
}
