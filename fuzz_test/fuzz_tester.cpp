#include <cstddef>
#include <cstdint>
#include <fmt/base.h>
#include <span>

[[nodiscard]] auto SumValues(const uint8_t *data, size_t size)
{
  constexpr auto kScale = 1000;

  int value = 0;
  for (const auto byte : std::span{ data, size }) { value += static_cast<int>(byte) * kScale; }
  return value;
}

// Fuzzer that attempts to invoke undefined behavior for signed integer overflow
// cppcheck-suppress unusedFunction symbolName=LLVMFuzzerTestOneInput
extern "C" auto LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) -> int
{
  fmt::print("Value sum: {}, len{}\n", SumValues(data, size), size);
  return 0;
}
