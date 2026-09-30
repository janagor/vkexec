#include <vkexec/error.hpp>

#include <vkexec/config.hpp>

#include <vulkan/vulkan_core.h>

#include <string>
#include <system_error>
#include <utility>

#ifdef VKEXEC_EXPECT_NO_EXCEPTIONS
static_assert(VKEXEC_HAS_EXCEPTIONS == 0);
#endif

namespace vkexec {

namespace {

  /**
   * Standard error category for `errc` values.
   *
   * @see category, make_error_code
   */
  class vkexec_error_category final : public std::error_category
  {
  public:
    // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
    [[nodiscard]] auto name() const noexcept -> char const * override { return "vkexec"; }

    // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
    [[nodiscard]] auto message(int error_value) const -> std::string override { return message_text(error_value); }

    // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
    [[nodiscard]] auto message_text(int error_value) const noexcept -> char const *
    {
      switch (static_cast<errc>(error_value)) {
      case errc::invalid_argument:
        return "invalid argument";
      case errc::io_error:
        return "I/O error";
      case errc::parse_error:
        return "parse error";
      case errc::unsupported:
        return "unsupported operation";
      case errc::out_of_range:
        return "out of range";
      case errc::empty_result:
        return "empty result";
      case errc::cancelled:
        return "cancelled";
      case errc::vulkan:
        return "vulkan error";
      case errc::unexpected_exception:
        return "unexpected C++ exception";
      default:
        return "unknown vkexec error";
      }
    }
  };

  /**
   * Standard error category whose values are failed `VkResult` integers.
   *
   * @see vulkan_category, make_vk_error_code
   */
  class vulkan_error_category final : public std::error_category
  {
  public:
    // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
    [[nodiscard]] auto name() const noexcept -> char const * override { return "vkexec.vulkan"; }

    // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
    [[nodiscard]] auto message(int error_value) const -> std::string override;

    // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
    [[nodiscard]] auto message_text(int error_value) const noexcept -> char const *;
  };

  auto vulkan_error_category::message(int error_value) const -> std::string { return message_text(error_value); }

  // cppcheck-suppress functionStatic
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  auto vulkan_error_category::message_text(int error_value) const noexcept -> char const *
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

}// namespace

auto category() noexcept -> std::error_category const &
{
  static vkexec_error_category const k_instance{};
  return k_instance;
}

auto vulkan_category() noexcept -> std::error_category const &
{
  static vulkan_error_category const k_instance{};
  return k_instance;
}

auto make_error_code(errc error) noexcept -> std::error_code
{ return std::error_code{ static_cast<int>(error), category() }; }

auto make_vk_error_code(int vk_result) noexcept -> std::error_code
{
  if (vk_result >= 0) { detail::contract_violation("make_vk_error_code requires a failed VkResult"); }
  return std::error_code{ vk_result, vulkan_category() };
}

auto make_error(errc code, std::string detail) -> error
{
  return error{
    .code = make_error_code(code),
    .detail = std::move(detail),
  };
}

auto to_string(error const &err) -> std::string
{
  if (err.detail.empty()) { return err.message(); }
  return err.detail;
}

}// namespace vkexec
