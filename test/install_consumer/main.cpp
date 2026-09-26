#include <vkexec/execution.hpp>

#include <stdexec/execution.hpp>

#include <type_traits>

int main()
{
  static_assert(!std::is_copy_constructible_v<vkexec::context>);
  return 0;
}
