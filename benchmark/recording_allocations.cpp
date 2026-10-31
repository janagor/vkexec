// Allocation instrumentation for this benchmark executable only. Tracks C++
// allocations on the recording thread; Vulkan driver mallocs are excluded.
#include <cstddef>
#include <cstdlib>
#include <new>
#ifdef _MSC_VER
#include <malloc.h>
#endif
#include <vkexec/config.hpp>
namespace recording_allocations {
thread_local bool enabled{};
thread_local std::size_t count{};
auto begin() noexcept -> void
{
  count = 0;
  enabled = true;
}
auto end() noexcept -> std::size_t
{
  enabled = false;
  return count;
}
}// namespace recording_allocations
namespace {
auto allocate(std::size_t size, std::size_t alignment = 0) -> void *
{
  if (recording_allocations::enabled) { ++recording_allocations::count; }
  size = size == 0 ? 1 : size;
#ifdef _MSC_VER
  void *ptr = alignment ? _aligned_malloc(size, alignment) : std::malloc(size);
#else
  void *ptr =
    alignment ? std::aligned_alloc(alignment, ((size + alignment - 1) / alignment) * alignment) : std::malloc(size);
#endif
  if (ptr) { return ptr; }
#if VKEXEC_HAS_EXCEPTIONS
  throw std::bad_alloc{};
#else
  std::abort();
#endif
}
}// namespace
auto operator new(std::size_t size) -> void * { return allocate(size); }
auto operator new[](std::size_t size) -> void * { return allocate(size); }
auto operator new(std::size_t size, std::align_val_t alignment) -> void *
{ return allocate(size, static_cast<std::size_t>(alignment)); }
auto operator new[](std::size_t size, std::align_val_t alignment) -> void *
{ return allocate(size, static_cast<std::size_t>(alignment)); }
auto operator delete(void *ptr) noexcept -> void { std::free(ptr); }
auto operator delete[](void *ptr) noexcept -> void { std::free(ptr); }
auto operator delete(void *ptr, std::size_t) noexcept -> void { std::free(ptr); }
auto operator delete[](void *ptr, std::size_t) noexcept -> void { std::free(ptr); }
auto operator delete(void *ptr, std::align_val_t) noexcept -> void
{
#ifdef _MSC_VER
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}
auto operator delete[](void *ptr, std::align_val_t alignment) noexcept -> void { ::operator delete(ptr, alignment); }
auto operator delete(void *ptr, std::size_t, std::align_val_t alignment) noexcept -> void
{ ::operator delete(ptr, alignment); }
auto operator delete[](void *ptr, std::size_t, std::align_val_t alignment) noexcept -> void
{ ::operator delete(ptr, alignment); }
