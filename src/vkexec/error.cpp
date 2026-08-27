#include <vkexec/error.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdlib>
#include <exception>
#include <string_view>

#ifdef BOOST_LEAF_NO_EXCEPTIONS
namespace boost {
[[noreturn]] void throw_exception(std::exception const &) { std::abort(); }
}// namespace boost
#endif

namespace vkexec {

auto vulkan_error_category::Message(int error_value) const noexcept -> std::string_view
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

}// namespace vkexec
