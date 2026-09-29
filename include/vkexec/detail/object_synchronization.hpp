#ifndef VKEXEC_DETAIL_OBJECT_SYNCHRONIZATION_HPP
#define VKEXEC_DETAIL_OBJECT_SYNCHRONIZATION_HPP

#include <vulkan/vulkan.h>

#include <vkexec/result.hpp>

#include <memory>
#include <mutex>
#include <unordered_map>

namespace vkexec {
class context;
}

namespace vkexec::detail {

// One state per actual queue handle, including aliases across context roles.
// Keep the lock mechanism here so a future adopted-queue policy can replace it.
class queue_synchronization_state
{
public:
  class guard
  {
  public:
    ~guard() = default;
    guard(guard const &) = delete;
    auto operator=(guard const &) -> guard & = delete;
    guard(guard &&) noexcept = default;
    auto operator=(guard &&) noexcept -> guard & = default;

  private:
    friend class queue_synchronization_state;
    explicit guard(std::mutex &mutex) : lock_(mutex) {}
    std::unique_lock<std::mutex> lock_;
  };

  [[nodiscard]] auto lock() -> guard { return guard{ mutex_ }; }

private:
  std::mutex mutex_;
};

template<class Handle, class State = std::mutex> class object_synchronization_registry
{
public:
  [[nodiscard]] auto state(Handle handle) -> std::shared_ptr<State>
  {
    std::scoped_lock const lock(mutex_);
    if (auto found = states_.find(handle); found != states_.end()) { return found->second; }
    auto created = std::make_shared<State>();
    states_.emplace(handle, created);
    return created;
  }

private:
  std::mutex mutex_;
  std::unordered_map<Handle, std::shared_ptr<State>> states_;
};

using queue_synchronization_registry = object_synchronization_registry<VkQueue, queue_synchronization_state>;
using descriptor_pool_synchronization_registry = object_synchronization_registry<VkDescriptorPool>;

// Pool states stay registered for the context lifetime. Reused handle values
// may share an old lock, which is conservative and avoids replacement races.
class descriptor_pool_access
{
public:
  [[nodiscard]] static auto lock(context const &ctx, VkDescriptorPool pool) -> std::unique_lock<std::mutex>;
  [[nodiscard]] static auto register_owned(context const &ctx, VkDescriptorPool pool) -> status;
};

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_OBJECT_SYNCHRONIZATION_HPP
