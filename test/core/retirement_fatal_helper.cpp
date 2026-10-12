#include "vkexec/detail/retirement_wait.hpp"

#include <cstdlib>
#include <exception>

namespace {

constexpr int k_fatal_exit_code = 73;

[[nodiscard]] auto VKAPI_CALL fail_retirement_wait(VkDevice /*device*/,
  std::uint32_t /*count*/,
  VkFence const * /*fences*/,
  VkBool32 /*wait_all*/,
  std::uint64_t /*timeout*/) -> VkResult
{ return VK_ERROR_OUT_OF_HOST_MEMORY; }

}// namespace

auto main() -> int
{
  std::set_terminate([]() -> void { std::_Exit(k_fatal_exit_code); });
  (void)vkexec::detail::wait_for_retirement(VK_NULL_HANDLE, 0, nullptr, fail_retirement_wait);
  return EXIT_FAILURE;
}
