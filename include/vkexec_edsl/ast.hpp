#ifndef VKEXEC_EDSL_AST_HPP
#define VKEXEC_EDSL_AST_HPP

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vkexec::edsl {

constexpr int k_default_local_size_x = 64;

enum class ValueType : std::uint8_t { Float, Int, Bool, Vec2, Vec3, Vec4 };

enum class OpKind : std::uint8_t {
  ConstFloat,
  ConstInt,
  ConstBool,
  Add,
  Sub,
  Mul,
  Div,
  Neg,
  Assign,
  Load,
  Store,
  Index,
  Var,
  ParamIndex,
  VertexIndex,
  PushField,
  InputVarying,
  OutputVarying,
  Select,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  Equal,
  NotEqual,
  LogicalAnd,
  LogicalOr,
  LogicalNot,
  Sin,
  Cos,
  Sqrt,
  Abs,
  Floor,
  Ceil,
  Min,
  Max,
  Vec2,
  Vec3,
  Vec4,
  Vec4From2,
  IfBegin,
  IfEnd,
  ElseBegin,
  ElseEnd,
  ForBegin,
  ForEnd,
  Statement
};

struct ExprNode
{
  OpKind kind{};
  ValueType type{ ValueType::Float };
  int lhs{ -1 };
  int rhs{ -1 };
  int extra{ -1 };
  int fourth{ -1 };
  double const_f{ 0.0 };
  std::int64_t const_i{ 0 };
  int binding{ -1 };
  std::string name;

  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  static auto make(OpKind kind, int left = -1, int right = -1, int third = -1, int fourth_comp = -1) -> ExprNode
  {
    ExprNode node;
    node.kind = kind;
    node.lhs = left;
    node.rhs = right;
    node.extra = third;
    node.fourth = fourth_comp;
    return node;
  }
  // NOLINTEND(bugprone-easily-swappable-parameters)
};

struct BufferBinding
{
  std::string name;
  std::string elem_glsl_type;
  int binding{ -1 };
  void *vk_buffer{ nullptr };
  std::size_t byte_size{ 0 };
  std::size_t elem_count{ 0 };
};

struct ASTContext
{
  std::vector<ExprNode> nodes;
  std::vector<BufferBinding> buffers;
  std::vector<std::string> statements;
  std::string push_block_glsl;
  std::size_t push_bytes{ 0 };
  int local_size_x{ k_default_local_size_x };
  int next_temp{ 0 };
  int next_binding{ 0 };

  auto append(ExprNode node) -> int
  {
    const int node_id = static_cast<int>(nodes.size());
    nodes.push_back(std::move(node));
    return node_id;
  }

  auto make_temp(std::string_view prefix = "t", ValueType type = ValueType::Float) -> int
  {
    const int temp_id = next_temp++;
    ExprNode node = ExprNode::make(OpKind::Var);
    node.name = std::string(prefix) + std::to_string(temp_id);
    node.type = type;
    return append(std::move(node));
  }
};

inline auto current_ast() noexcept -> ASTContext *&
{
  thread_local ASTContext *ctx = nullptr;
  return ctx;
}

class ASTScope
{
  ASTContext *previous_;

public:
  explicit ASTScope(ASTContext &ctx) : previous_(current_ast()) { current_ast() = &ctx; }
  ~ASTScope() { current_ast() = previous_; }
  ASTScope(const ASTScope &) = delete;
  auto operator=(const ASTScope &) -> ASTScope & = delete;
  ASTScope(ASTScope &&) = delete;
  auto operator=(ASTScope &&) -> ASTScope & = delete;
};

inline auto ast() -> ASTContext &
{
  ASTContext *ctx = current_ast();
  if (ctx == nullptr) { throw std::runtime_error("vkexec::edsl AST used outside of a tracing scope"); }
  return *ctx;
}

inline auto glsl_type_name(ValueType type) -> const char *
{
  switch (type) {
  case ValueType::Int:
    return "int";
  case ValueType::Bool:
    return "bool";
  case ValueType::Vec2:
    return "vec2";
  case ValueType::Vec3:
    return "vec3";
  case ValueType::Vec4:
    return "vec4";
  case ValueType::Float:
  default:
    return "float";
  }
}

}// namespace vkexec::edsl

#endif// VKEXEC_EDSL_AST_HPP
