#ifndef VKEXEC_ERROR_HPP
#define VKEXEC_ERROR_HPP

#include <cx_system_error/system_error.hpp>

// Pedantic clang rejects LEAF's GNU stmt-expr BOOST_LEAF_CHECK; use the portable form.
#ifndef BOOST_LEAF_CFG_GNUC_STMTEXPR
#define BOOST_LEAF_CFG_GNUC_STMTEXPR 0
#endif
#include <boost/leaf.hpp>

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace vkexec {

namespace leaf = boost::leaf;

// NOLINTNEXTLINE(performance-enum-size)
enum class errc {
  invalid_argument = 1,
  io_error,
  parse_error,
  unsupported,
  out_of_range,
  empty_result,
  cancelled,
  vulkan,
};

class vkexec_error_category final : public cx::ErrorCategory
{
public:
  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] constexpr auto Name() const noexcept -> char const * override { return "vkexec"; }

  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] constexpr auto Message(int error_value) const noexcept -> std::string_view override
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
    default:
      return "unknown vkexec error";
    }
  }
};

class vulkan_error_category final : public cx::ErrorCategory
{
public:
  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] constexpr auto Name() const noexcept -> char const * override { return "vkexec.vulkan"; }

  // NOLINTNEXTLINE(readability-identifier-naming,readability-convert-member-functions-to-static)
  [[nodiscard]] auto Message(int error_value) const noexcept -> std::string_view override;
};

[[nodiscard]] inline auto category() noexcept -> cx::ErrorCategory const &
{
  static vkexec_error_category const k_instance{};
  return k_instance;
}

[[nodiscard]] inline auto vulkan_category() noexcept -> cx::ErrorCategory const &
{
  static vulkan_error_category const k_instance{};
  return k_instance;
}

// NOLINTNEXTLINE(readability-identifier-naming)
[[nodiscard]] inline auto MakeErrorCode(errc error) noexcept -> cx::ErrorCode
{ return cx::ErrorCode{ static_cast<int>(error), category() }; }

// NOLINTNEXTLINE(readability-identifier-naming)
[[nodiscard]] inline auto MakeVkErrorCode(int vk_result) noexcept -> cx::ErrorCode
{ return cx::ErrorCode{ vk_result, vulkan_category() }; }

struct error
{
  cx::ErrorCode code{};
  std::string detail;

  [[nodiscard]] auto message() const noexcept -> std::string_view
  { return detail.empty() ? code.Message() : std::string_view(detail); }
};

template<typename T> using result = leaf::result<T>;
using status = leaf::result<void>;

namespace detail {

  struct last_error_slot
  {
    int id{ 0 };
    error value{};
  };

  // LEAF drops e-types when no context slot is active; stash the payload for to_error.
  [[nodiscard]] inline auto last_error() noexcept -> last_error_slot &
  {
    thread_local last_error_slot slot{};
    return slot;
  }

  [[nodiscard]] inline auto stash_error(error err) -> leaf::error_id
  {
    leaf::error_id const id = leaf::new_error(err);
    last_error() = { .id = id.value(), .value = std::move(err) };
    return id;
  }

}// namespace detail

[[nodiscard]] inline auto make_error(errc code, std::string detail = {}) -> leaf::error_id
{
  return detail::stash_error(error{
    .code = MakeErrorCode(code),
    .detail = std::move(detail),
  });
}

/// Load a `vkexec::error` previously attached to `id` (for sender set_error bridging).
[[nodiscard]] inline auto to_error(leaf::error_id error_id) -> error
{
  auto const &slot = detail::last_error();
  if (slot.id == error_id.value()) { return slot.value; }

  error err{ .code = MakeErrorCode(errc::unsupported), .detail = "unknown error" };
  leaf::try_handle_all([&]() -> leaf::result<void> { return error_id; },
    [&](error loaded) -> void { err = std::move(loaded); },
    []() -> void {});
  return err;
}

[[nodiscard]] inline auto to_string(error const &err) -> std::string
{
  if (err.detail.empty()) { return std::string(err.code.Message()); }
  return err.detail;
}

}// namespace vkexec

namespace cx {

// NOLINTNEXTLINE(readability-identifier-naming)
template<> struct IsErrorCodeEnum<vkexec::errc> : std::true_type
{
};

}// namespace cx

#endif// VKEXEC_ERROR_HPP
