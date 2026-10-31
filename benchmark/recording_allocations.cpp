// Allocation instrumentation for this benchmark executable only. Tracks C++
// allocations on the recording thread; Vulkan driver mallocs are excluded.
#include "recording_allocations.hpp"

#include <cstddef>
#include <cstdlib>
#include <new>
#include <vkexec/config.hpp>

#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace {
constexpr std::size_t k_zero_size = 0;
constexpr std::size_t k_no_alignment = 0;
constexpr std::size_t k_nonzero_allocation_size = 1;
constexpr std::size_t k_no_padding = 1;
struct allocation_tracking
{
  bool enabled{};
  std::size_t count{};
};

auto tracking_state() noexcept -> allocation_tracking &
{
  thread_local allocation_tracking state;
  return state;
}
}// namespace

namespace recording_allocations {
auto begin() noexcept -> void
{
  auto &state = tracking_state();
  state.count = 0;
  state.enabled = true;
}
auto end() noexcept -> std::size_t
{
  auto &state = tracking_state();
  state.enabled = false;
  return state.count;
}
}// namespace recording_allocations
namespace {
auto allocate(std::size_t size, std::size_t alignment = k_no_alignment) -> void *
{
  auto &state = tracking_state();
  if (state.enabled) { ++state.count; }
  size = size == k_zero_size ? k_nonzero_allocation_size : size;
#ifdef _MSC_VER
  // Global allocation replacements must use the C runtime allocation API.
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,hicpp-no-malloc,cppcoreguidelines-owning-memory)
  void *ptr = alignment != k_no_alignment ? _aligned_malloc(size, alignment) : std::malloc(size);
#else
  void *ptr{};
  if (alignment != k_no_alignment) {
    // Global allocation replacements must use the C runtime allocation API.
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,hicpp-no-malloc,cppcoreguidelines-owning-memory)
    ptr = std::aligned_alloc(alignment, ((size + alignment - k_no_padding) / alignment) * alignment);
  } else {
    // Global allocation replacements must use the C runtime allocation API.
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,hicpp-no-malloc,cppcoreguidelines-owning-memory)
    ptr = std::malloc(size);
  }
#endif
  if (ptr != nullptr) { return ptr; }
#if VKEXEC_HAS_EXCEPTIONS
  throw std::bad_alloc{};
#else
  std::abort();
#endif
}

auto deallocate(void *ptr, std::size_t alignment = k_no_alignment) noexcept -> void
{
#ifdef _MSC_VER
  if (alignment != k_no_alignment) {
    _aligned_free(ptr);
    return;
  }
#else
  (void)alignment;
#endif
  // This is the release boundary for memory obtained by the global new replacements.
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,hicpp-no-malloc,cppcoreguidelines-owning-memory)
  std::free(ptr);
}
}// namespace
auto operator new(std::size_t size) -> void * { return allocate(size); }
auto operator new[](std::size_t size) -> void * { return allocate(size); }
auto operator new(std::size_t size, std::align_val_t alignment) -> void *
{ return allocate(size, static_cast<std::size_t>(alignment)); }
auto operator new[](std::size_t size, std::align_val_t alignment) -> void *
{ return allocate(size, static_cast<std::size_t>(alignment)); }
auto operator delete(void *ptr) noexcept -> void { deallocate(ptr); }
auto operator delete[](void *ptr) noexcept -> void { deallocate(ptr); }
auto operator delete(void *ptr, std::size_t size) noexcept -> void
{
  (void)size;
  deallocate(ptr);
}
auto operator delete[](void *ptr, std::size_t size) noexcept -> void
{
  (void)size;
  deallocate(ptr);
}
auto operator delete(void *ptr, std::align_val_t alignment) noexcept -> void
{ deallocate(ptr, static_cast<std::size_t>(alignment)); }
auto operator delete[](void *ptr, std::align_val_t alignment) noexcept -> void
{ deallocate(ptr, static_cast<std::size_t>(alignment)); }
auto operator delete(void *ptr, std::size_t size, std::align_val_t alignment) noexcept -> void
{
  (void)size;
  deallocate(ptr, static_cast<std::size_t>(alignment));
}
auto operator delete[](void *ptr, std::size_t size, std::align_val_t alignment) noexcept -> void
{
  (void)size;
  deallocate(ptr, static_cast<std::size_t>(alignment));
}
