#ifndef VKEXEC_EXTENSIONS_COMMON_PROC_CACHE_HPP
#define VKEXEC_EXTENSIONS_COMMON_PROC_CACHE_HPP

//! \file
//! Per-device cached extension function-pointer tables.

#include <vulkan/vulkan.h>

#include <mutex>
#include <unordered_map>
#include <utility>

namespace vkexec::ext::detail {

/**
 * Returns a cached `Procs` table for `device`, loading it once via `load`.
 *
 * Thread-safe. The returned reference is stable for the process lifetime of the
 * cache entry (devices are not currently purged).
 *
 * @param device Logical device key.
 * @param load Callable `Procs(VkDevice)` that resolves PFNs.
 */
template<typename Procs, typename Loader> [[nodiscard]] auto cached_procs(VkDevice device, Loader load) -> Procs const &
{
  static std::mutex mutex;
  static std::unordered_map<VkDevice, Procs> cache;
  std::scoped_lock const lock(mutex);
  auto const found = cache.find(device);
  if (found != cache.end()) { return found->second; }
  return cache.emplace(device, load(device)).first->second;
}

}// namespace vkexec::ext::detail

#endif// VKEXEC_EXTENSIONS_COMMON_PROC_CACHE_HPP
