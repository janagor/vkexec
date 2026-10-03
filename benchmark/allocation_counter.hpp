#ifndef VKEXEC_BENCHMARK_ALLOCATION_COUNTER_HPP
#define VKEXEC_BENCHMARK_ALLOCATION_COUNTER_HPP

#include <cstddef>

namespace vkexec_benchmark {

struct allocation_counts
{
  std::size_t allocations{};
  std::size_t bytes{};
};

// Counts allocations on the benchmark thread while a graph is built.
class allocation_counter
{
public:
  allocation_counter() noexcept;
  ~allocation_counter();

  allocation_counter(allocation_counter const &) = delete;
  auto operator=(allocation_counter const &) -> allocation_counter & = delete;

  [[nodiscard]] auto counts() const noexcept -> allocation_counts;

private:
  allocation_counts counts_{};
  allocation_counts *previous_{};
};

}// namespace vkexec_benchmark

#endif// VKEXEC_BENCHMARK_ALLOCATION_COUNTER_HPP
