#include <myproject/sample_library.hpp>

auto Factorial(int input) noexcept -> int
{
  int result = 1;

  while (input > 0) {
    result *= input;
    --input;
  }

  return result;
}
