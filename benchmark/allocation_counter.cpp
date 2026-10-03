#include "allocation_counter.hpp"

#include <vkexec/config.hpp>

#include <cstdlib>
#include <new>

namespace {
thread_local vkexec_benchmark::allocation_counts *active_counts = nullptr;

auto allocate(std::size_t size) -> void *
{
  if (size == 0) { size = 1; }
  void *ptr = std::malloc(size);
  if (ptr == nullptr) {
#if VKEXEC_HAS_EXCEPTIONS
    throw std::bad_alloc{};
#else
    std::abort();
#endif
  }
  if (active_counts != nullptr) {
    ++active_counts->allocations;
    active_counts->bytes += size;
  }
  return ptr;
}
}// namespace

auto operator new(std::size_t size) -> void * { return allocate(size); }
auto operator new[](std::size_t size) -> void * { return allocate(size); }
auto operator delete(void *ptr) noexcept -> void { std::free(ptr); }
auto operator delete[](void *ptr) noexcept -> void { std::free(ptr); }
auto operator delete(void *ptr, std::size_t /*size*/) noexcept -> void { std::free(ptr); }
auto operator delete[](void *ptr, std::size_t /*size*/) noexcept -> void { std::free(ptr); }

namespace vkexec_benchmark {

allocation_counter::allocation_counter() noexcept : previous_(active_counts) { active_counts = &counts_; }
allocation_counter::~allocation_counter() { active_counts = previous_; }
auto allocation_counter::counts() const noexcept -> allocation_counts { return counts_; }

}// namespace vkexec_benchmark
