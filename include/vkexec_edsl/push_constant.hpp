#ifndef VKEXEC_EDSL_PUSH_CONSTANT_HPP
#define VKEXEC_EDSL_PUSH_CONSTANT_HPP

#include <vkexec_edsl/types.hpp>

#include <boost/describe/class.hpp>
#include <boost/describe/members.hpp>
#include <boost/describe/modifiers.hpp>
#include <boost/mp11/algorithm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace vkexec::edsl {
namespace detail {

template<typename T>
inline auto glsl_type_name_of() -> const char *
{
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    return "float";
  } else if constexpr (std::is_same_v<T, bool>) {
    return "bool";
  } else if constexpr (std::is_unsigned_v<T>) {
    return "uint";
  } else if constexpr (std::is_integral_v<T>) {
    return "int";
  } else {
    static_assert(sizeof(T) == 0, "unsupported push-constant member type");
    return "float";
  }
}

template<typename T>
using describe_members_t = boost::describe::describe_members<T, boost::describe::mod_any_access>;

template<typename T>
inline constexpr std::size_t k_describe_member_count = boost::mp11::mp_size<describe_members_t<T>>::value;

template<typename MemberT>
using proxy_type_for = std::conditional_t<std::is_integral_v<MemberT> && !std::is_same_v<MemberT, bool>, Int, Float>;

template<typename T>
auto member_byte_offset(auto member_pointer) -> std::int64_t
{
  alignas(T) std::array<unsigned char, sizeof(T)> storage{};
  auto *object = reinterpret_cast<T *>(storage.data()); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
  auto *member = reinterpret_cast<unsigned char *>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    std::addressof(object->*member_pointer));
  return static_cast<std::int64_t>(member - storage.data());
}

template<auto MemberPtr>
struct matches_member_pointer {
  template<typename Descriptor>
  using fn = std::bool_constant<Descriptor::pointer == MemberPtr>;
};

} // namespace detail

/// Tracing proxy for a Boost.Describe'd POD push-constant struct.
///
/// Describe the host struct, then access fields with `pc.get<&Params::field>()`:
/// ```
/// struct params { float dt; };
/// BOOST_DESCRIBE_STRUCT(params, (), (dt))
/// ...
/// auto dt = pc.get<&params::dt>();
/// ```
template<typename T>
struct push_constant {
  static constexpr std::size_t k_byte_size = sizeof(T);
  static constexpr std::size_t k_member_count = detail::k_describe_member_count<T>;

  std::array<int, k_member_count == 0 ? 1 : k_member_count> field_ids{};

  [[nodiscard]] static auto glsl_block() -> std::string
  {
    std::string body;
    boost::mp11::mp_for_each<detail::describe_members_t<T>>([&](auto descriptor) -> void {
      using member_type =
        std::remove_cv_t<std::remove_reference_t<decltype(std::declval<T>().*descriptor.pointer)>>;
      body += "  ";
      body += detail::glsl_type_name_of<member_type>();
      body += ' ';
      body += descriptor.name;
      body += ";\n";
    });
    return "layout(push_constant) uniform PushConstants {\n" + body + "} pc;\n";
  }

  [[nodiscard]] static auto bind() -> push_constant
  {
    static_assert(k_member_count > 0, "push_constant<T> requires BOOST_DESCRIBE_STRUCT(T, (), (members...))");
    push_constant proxy{};
    std::size_t index = 0;
    boost::mp11::mp_for_each<detail::describe_members_t<T>>([&](auto descriptor) -> void {
      ExprNode node = ExprNode::make(OpKind::PushField);
      node.name = descriptor.name;
      node.const_i = detail::member_byte_offset<T>(descriptor.pointer);
      proxy.field_ids.at(index) = ast().append(std::move(node));
      ++index;
    });
    ast().push_block_glsl = glsl_block();
    ast().push_bytes = k_byte_size;
    return proxy;
  }

  /// Look up the tracing handle for a described data member.
  template<auto MemberPtr>
  [[nodiscard]] auto get() const
  {
    using member_type = std::remove_cv_t<std::remove_reference_t<decltype(std::declval<T>().*MemberPtr)>>;
    using index_t = boost::mp11::mp_find_if_q<detail::describe_members_t<T>, detail::matches_member_pointer<MemberPtr>>;
    static_assert(index_t::value < k_member_count, "member pointer is not described for this push_constant type");
    return detail::proxy_type_for<member_type>{ field_ids.at(index_t::value) };
  }
};

} // namespace vkexec::edsl

#endif // VKEXEC_EDSL_PUSH_CONSTANT_HPP
