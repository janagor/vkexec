#ifndef SAMPLE_LIBRARY_HPP
#define SAMPLE_LIBRARY_HPP

#include <myproject/sample_library_export.hpp>

[[nodiscard]] SAMPLE_LIBRARY_EXPORT auto Factorial(int input) noexcept -> int;

[[nodiscard]] constexpr auto FactorialConstexpr(int input) noexcept -> int
{
  if (input == 0) { return 1; }

  return input * FactorialConstexpr(input - 1);
}

#endif
