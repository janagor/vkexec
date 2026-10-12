#include "detail/test_vulkan_dispatch.hpp"

#include <atomic>
#include <exception>

namespace vkexec::detail {
namespace {
  std::atomic<std::shared_ptr<test_vulkan_dispatch const>> active_dispatch{};
}// namespace

auto set_test_vulkan_dispatch(test_vulkan_dispatch const *dispatch) -> void
{
  if (dispatch == nullptr) {
    active_dispatch.store({}, std::memory_order_release);
    return;
  }
  std::shared_ptr<test_vulkan_dispatch const> expected;
  auto snapshot = std::make_shared<test_vulkan_dispatch>(*dispatch);
  if (!active_dispatch.compare_exchange_strong(expected, std::move(snapshot), std::memory_order_acq_rel)) {
    std::terminate();
  }
}

auto current_test_vulkan_dispatch() noexcept -> std::shared_ptr<test_vulkan_dispatch const>
{ return active_dispatch.load(std::memory_order_acquire); }

}// namespace vkexec::detail
