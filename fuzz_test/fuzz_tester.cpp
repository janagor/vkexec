#include <cstddef>
#include <cstdint>
#include <fmt/base.h>
#include <span>

[[nodiscard]] auto sum_values(const uint8_t *data, size_t size) -> int
{
  constexpr auto k_scale = 1000;

  int value = 0;
  for (const auto byte : std::span{ data, size }) { value += static_cast<int>(byte) * k_scale; }
  return value;
}

// Fuzzer that attempts to invoke undefined behavior for signed integer overflow
// cppcheck-suppress unusedFunction symbolName=LLVMFuzzerTestOneInput
// NOLINTNEXTLINE(readability-identifier-naming)
extern "C" auto LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) -> int
{
  fmt::print("Value sum: {}, len{}\n", sum_values(data, size), size);
  return 0;
}
