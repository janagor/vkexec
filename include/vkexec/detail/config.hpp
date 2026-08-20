#ifndef VKEXEC_DETAIL_CONFIG_HPP
#define VKEXEC_DETAIL_CONFIG_HPP

#include <exception>

namespace vkexec::detail {

[[noreturn]] inline auto terminate() noexcept -> void { std::terminate(); }

struct catch_any_lvalue_t
{
  template<typename T>
  // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
  constexpr operator T &() const noexcept;
};

inline constexpr catch_any_lvalue_t k_catch_any_lvalue{};

}// namespace vkexec::detail

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#if defined(_MSC_VER) || defined(__clang__) && defined(_MSC_VER)
#define VKEXEC_NO_STDCPP_EXCEPTIONS() (_HAS_EXCEPTIONS == 0) || (_CPPUNWIND == 0)
#else
#define VKEXEC_NO_STDCPP_EXCEPTIONS() (__EXCEPTIONS == 0)
#endif

#if VKEXEC_NO_STDCPP_EXCEPTIONS()
#define VKEXEC_TRY if constexpr (true) {
#define VKEXEC_CATCH(...)                                                       \
  }                                                                             \
  else if constexpr (__VA_ARGS__ = ::vkexec::detail::k_catch_any_lvalue; false) \
  {
#define VKEXEC_CATCH_ALL      \
  }                           \
  else if constexpr (true) {} \
  else
#define VKEXEC_THROW(...) ::vkexec::detail::terminate()
#define VKEXEC_CATCH_FALLTHROUGH \
  }                              \
  else {}
#else
#define VKEXEC_TRY try
#define VKEXEC_CATCH catch
#define VKEXEC_CATCH_ALL catch (...)
#define VKEXEC_THROW(...) throw __VA_ARGS__
#define VKEXEC_CATCH_FALLTHROUGH
#endif
// NOLINTEND(cppcoreguidelines-macro-usage)

#endif// VKEXEC_DETAIL_CONFIG_HPP
