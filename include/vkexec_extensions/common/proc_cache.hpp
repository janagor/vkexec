#ifndef VKEXEC_EXTENSIONS_COMMON_PROC_CACHE_HPP
#define VKEXEC_EXTENSIONS_COMMON_PROC_CACHE_HPP

#include <vulkan/vulkan.h>

#include <mutex>
#include <unordered_map>
#include <utility>

namespace vkexec::ext::detail {

template<typename Procs, typename Loader>
[[nodiscard]] auto cached_procs(VkDevice device, Loader load) -> Procs const &
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
