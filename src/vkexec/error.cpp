#include <vkexec/error.hpp>

#include <boost/leaf/error.hpp>
#include <boost/leaf/handle_errors.hpp>
#include <boost/leaf/result.hpp>
#include <boost/system/detail/error_category.hpp>
#include <boost/system/detail/error_code.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdlib>
#include <exception>
#include <string>
#include <utility>

#ifdef BOOST_LEAF_NO_EXCEPTIONS
namespace boost {
[[noreturn]] void throw_exception(std::exception const & /*exception*/) { std::abort(); }
}// namespace boost
#endif

namespace vkexec {

auto vulkan_error_category::message(int error_value) const -> std::string
{
  return message(error_value, nullptr, 0);
}

auto vulkan_error_category::message(int error_value, char * /*buffer*/, std::size_t /*len*/) const noexcept
  -> char const *
{
  switch (static_cast<VkResult>(error_value)) {
  case VK_SUCCESS:
    return "success";
  case VK_NOT_READY:
    return "not ready";
  case VK_TIMEOUT:
    return "timeout";
  case VK_EVENT_SET:
    return "event set";
  case VK_EVENT_RESET:
    return "event reset";
  case VK_INCOMPLETE:
    return "incomplete";
  case VK_ERROR_OUT_OF_HOST_MEMORY:
    return "out of host memory";
  case VK_ERROR_OUT_OF_DEVICE_MEMORY:
    return "out of device memory";
  case VK_ERROR_INITIALIZATION_FAILED:
    return "initialization failed";
  case VK_ERROR_DEVICE_LOST:
    return "device lost";
  case VK_ERROR_MEMORY_MAP_FAILED:
    return "memory map failed";
  case VK_ERROR_LAYER_NOT_PRESENT:
    return "layer not present";
  case VK_ERROR_EXTENSION_NOT_PRESENT:
    return "extension not present";
  case VK_ERROR_FEATURE_NOT_PRESENT:
    return "feature not present";
  case VK_ERROR_INCOMPATIBLE_DRIVER:
    return "incompatible driver";
  case VK_ERROR_TOO_MANY_OBJECTS:
    return "too many objects";
  case VK_ERROR_FORMAT_NOT_SUPPORTED:
    return "format not supported";
  case VK_ERROR_FRAGMENTED_POOL:
    return "fragmented pool";
  case VK_ERROR_UNKNOWN:
    return "unknown error";
  case VK_ERROR_OUT_OF_POOL_MEMORY:
    return "out of pool memory";
  case VK_ERROR_INVALID_EXTERNAL_HANDLE:
    return "invalid external handle";
  case VK_ERROR_FRAGMENTATION:
    return "fragmentation";
  case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS:
    return "invalid opaque capture address";
  case VK_ERROR_SURFACE_LOST_KHR:
    return "surface lost";
  case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
    return "native window in use";
  case VK_SUBOPTIMAL_KHR:
    return "suboptimal";
  case VK_ERROR_OUT_OF_DATE_KHR:
    return "out of date";
  case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
    return "incompatible display";
  case VK_ERROR_VALIDATION_FAILED_EXT:
    return "validation failed";
  default:
    return "vulkan error";
  }
}

auto category() noexcept -> sys::error_category const &
{
  static vkexec_error_category const k_instance{};
  return k_instance;
}

auto vulkan_category() noexcept -> sys::error_category const &
{
  static vulkan_error_category const k_instance{};
  return k_instance;
}

auto make_error_code(errc error) noexcept -> sys::error_code
{
  return sys::error_code{ static_cast<int>(error), category() };
}

auto make_vk_error_code(int vk_result) noexcept -> sys::error_code
{
  return sys::error_code{ vk_result, vulkan_category() };
}

namespace detail {

auto last_error() noexcept -> last_error_slot &
{
  thread_local last_error_slot slot{};
  return slot;
}

auto stash_error(error err) -> leaf::error_id
{
  leaf::error_id const leaf_id = leaf::new_error(err);
  last_error() = { .id = leaf_id.value(), .value = std::move(err) };
  return leaf_id;
}

}// namespace detail

auto make_error(errc code, std::string detail) -> leaf::error_id
{
  return detail::stash_error(error{
    .code = make_error_code(code),
    .detail = std::move(detail),
  });
}

auto to_error(leaf::error_id error_id) -> error
{
  auto const &slot = detail::last_error();
  if (slot.id == error_id.value()) { return slot.value; }

  error err{ .code = make_error_code(errc::unsupported), .detail = "unknown error" };
  leaf::try_handle_all([&]() -> leaf::result<void> { return error_id; },
    [&](error loaded) -> void { err = std::move(loaded); },
    []() -> void {});
  return err;
}

auto to_string(error const &err) -> std::string
{
  if (err.detail.empty()) { return std::string(err.message()); }
  return err.detail;
}

}// namespace vkexec
