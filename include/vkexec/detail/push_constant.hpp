#ifndef VKEXEC_DETAIL_PUSH_CONSTANT_HPP
#define VKEXEC_DETAIL_PUSH_CONSTANT_HPP


#include <vkexec/detail/types.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace vlk {

/// GPU push-constant proxy. Member access is provided by VLK_PUSH_CONSTANT.
template<typename T>
struct PushConstant;

namespace detail {

inline auto glsl_type_name(float tag) -> const char *
{
  (void)tag;
  return "float";
}
inline auto glsl_type_name(double tag) -> const char *
{
  (void)tag;
  return "float";
}
inline auto glsl_type_name(int tag) -> const char *
{
  (void)tag;
  return "int";
}
inline auto glsl_type_name(unsigned tag) -> const char *
{
  (void)tag;
  return "uint";
}
inline auto glsl_type_name(bool tag) -> const char *
{
  (void)tag;
  return "bool";
}

} // namespace detail
} // namespace vlk

// Reflection macros intentionally emit struct specializations and field lists.
// NOLINTBEGIN(cppcoreguidelines-macro-usage,bugprone-macro-parentheses,hicpp-vararg)

/// Reflect a POD push-constant struct into a tracing proxy.
/// Usage: `VLK_PUSH_CONSTANT(SimParams, float, dt, float, damping)`
#define VLK_PUSH_CONSTANT(Type, ...)                                                                                   \
  template<>                                                                                                           \
  struct vlk::PushConstant<Type> {                                                                                     \
    VLK_PC_DECL_MEMBERS(__VA_ARGS__)                                                                                   \
    static constexpr std::size_t byte_size = sizeof(Type);                                                             \
    static std::string glsl_block()                                                                                    \
    {                                                                                                                  \
      std::string body;                                                                                                \
      VLK_PC_EMIT_GLSL(__VA_ARGS__)                                                                                    \
      return "layout(push_constant) uniform PushConstants {\n" + body + "} pc;\n";                                     \
    }                                                                                                                  \
    static PushConstant bind()                                                                                         \
    {                                                                                                                  \
      PushConstant proxy{};                                                                                            \
      VLK_PC_BIND_FIELDS(Type, __VA_ARGS__)                                                                            \
      vlk::ast().push_block_glsl = glsl_block();                                                                       \
      vlk::ast().push_bytes = byte_size;                                                                               \
      return proxy;                                                                                                    \
    }                                                                                                                  \
  }

#define VLK_PC_FIELD_TYPE(type)                                                                                        \
  std::conditional_t<std::is_integral_v<type> && !std::is_same_v<type, bool>, vlk::Int, vlk::Float>

#define VLK_PC_DECL_1(t1, n1) VLK_PC_FIELD_TYPE(t1) n1;
#define VLK_PC_DECL_2(t1, n1, t2, n2)                                                                                  \
  VLK_PC_DECL_1(t1, n1)                                                                                                \
  VLK_PC_DECL_1(t2, n2)
#define VLK_PC_DECL_3(t1, n1, t2, n2, t3, n3)                                                                          \
  VLK_PC_DECL_2(t1, n1, t2, n2)                                                                                        \
  VLK_PC_DECL_1(t3, n3)
#define VLK_PC_DECL_4(t1, n1, t2, n2, t3, n3, t4, n4)                                                                  \
  VLK_PC_DECL_3(t1, n1, t2, n2, t3, n3)                                                                                \
  VLK_PC_DECL_1(t4, n4)

#define VLK_PC_DECL_MEMBERS(...) VLK_PC_DECL_MEMBERS_N(VLK_PC_NARG(__VA_ARGS__), __VA_ARGS__)
#define VLK_PC_DECL_MEMBERS_N(N, ...) VLK_PC_DECL_MEMBERS_N_(N, __VA_ARGS__)
#define VLK_PC_DECL_MEMBERS_N_(N, ...) VLK_PC_DECL_##N(__VA_ARGS__)

#define VLK_PC_GLSL_1(t1, n1) body += std::string("  ") + vlk::detail::glsl_type_name(t1{}) + " " #n1 ";\n";
#define VLK_PC_GLSL_2(t1, n1, t2, n2)                                                                                  \
  VLK_PC_GLSL_1(t1, n1)                                                                                                \
  VLK_PC_GLSL_1(t2, n2)
#define VLK_PC_GLSL_3(t1, n1, t2, n2, t3, n3)                                                                          \
  VLK_PC_GLSL_2(t1, n1, t2, n2)                                                                                        \
  VLK_PC_GLSL_1(t3, n3)
#define VLK_PC_GLSL_4(t1, n1, t2, n2, t3, n3, t4, n4)                                                                  \
  VLK_PC_GLSL_3(t1, n1, t2, n2, t3, n3)                                                                                \
  VLK_PC_GLSL_1(t4, n4)

#define VLK_PC_EMIT_GLSL(...) VLK_PC_EMIT_GLSL_N(VLK_PC_NARG(__VA_ARGS__), __VA_ARGS__)
#define VLK_PC_EMIT_GLSL_N(N, ...) VLK_PC_EMIT_GLSL_N_(N, __VA_ARGS__)
#define VLK_PC_EMIT_GLSL_N_(N, ...) VLK_PC_GLSL_##N(__VA_ARGS__)

#define VLK_PC_BIND_ONE(Type, type, field)                                                                             \
  do {                                                                                                                 \
    vlk::ExprNode node = vlk::ExprNode::make(vlk::OpKind::PushField);                                                  \
    node.name = #field;                                                                                                \
    node.const_i = static_cast<std::int64_t>(offsetof(Type, field));                                                   \
    /* Set .id directly — Float/Int operator= emits AST assigns and must not run during bind. */                       \
    proxy.field.id = vlk::ast().append(std::move(node));                                                               \
  } while (0)

#define VLK_PC_BIND_1(Type, t1, n1) VLK_PC_BIND_ONE(Type, t1, n1);
#define VLK_PC_BIND_2(Type, t1, n1, t2, n2)                                                                            \
  VLK_PC_BIND_1(Type, t1, n1)                                                                                          \
  VLK_PC_BIND_1(Type, t2, n2)
#define VLK_PC_BIND_3(Type, t1, n1, t2, n2, t3, n3)                                                                    \
  VLK_PC_BIND_2(Type, t1, n1, t2, n2)                                                                                  \
  VLK_PC_BIND_1(Type, t3, n3)
#define VLK_PC_BIND_4(Type, t1, n1, t2, n2, t3, n3, t4, n4)                                                            \
  VLK_PC_BIND_3(Type, t1, n1, t2, n2, t3, n3)                                                                          \
  VLK_PC_BIND_1(Type, t4, n4)

#define VLK_PC_BIND_FIELDS(Type, ...) VLK_PC_BIND_FIELDS_N(Type, VLK_PC_NARG(__VA_ARGS__), __VA_ARGS__)
#define VLK_PC_BIND_FIELDS_N(Type, N, ...) VLK_PC_BIND_FIELDS_N_(Type, N, __VA_ARGS__)
#define VLK_PC_BIND_FIELDS_N_(Type, N, ...) VLK_PC_BIND_##N(Type, __VA_ARGS__)

// Number of (type, name) pairs from a flat type,name,... list (2/4/6/8 args → 1/2/3/4 pairs)
#define VLK_PC_NARG(...) VLK_PC_NARG_(__VA_ARGS__, 4, 4, 3, 3, 2, 2, 1, 1, 0)
#define VLK_PC_NARG_(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N

// NOLINTEND(cppcoreguidelines-macro-usage,bugprone-macro-parentheses,hicpp-vararg)

#endif  // VKEXEC_DETAIL_PUSH_CONSTANT_HPP
