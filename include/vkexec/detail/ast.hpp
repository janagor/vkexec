#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vlk {

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
  PushField,
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
  IfBegin,
  IfEnd,
  ElseBegin,
  ElseEnd,
  ForBegin,
  ForEnd,
  Statement
};

struct ExprNode {
  OpKind kind{};
  int lhs{ -1 };
  int rhs{ -1 };
  int extra{ -1 };
  double const_f{ 0.0 };
  std::int64_t const_i{ 0 };
  int binding{ -1 };
  std::string name;

  static ExprNode make(OpKind k, int left = -1, int right = -1, int ext = -1)
  {
    ExprNode n;
    n.kind = k;
    n.lhs = left;
    n.rhs = right;
    n.extra = ext;
    return n;
  }
};

struct BufferBinding {
  std::string name;
  std::string elem_glsl_type;
  int binding{ -1 };
  void *vk_buffer{ nullptr };
  std::size_t byte_size{ 0 };
  std::size_t elem_count{ 0 };
};

struct ASTContext {
  std::vector<ExprNode> nodes;
  std::vector<BufferBinding> buffers;
  std::vector<std::string> statements;
  std::string push_block_glsl;
  std::size_t push_bytes{ 0 };
  int local_size_x{ 64 };
  int next_temp{ 0 };
  int next_binding{ 0 };

  int append(ExprNode node)
  {
    const int id = static_cast<int>(nodes.size());
    nodes.push_back(std::move(node));
    return id;
  }

  int make_temp(std::string_view prefix = "t")
  {
    const int id = next_temp++;
    ExprNode n = ExprNode::make(OpKind::Var);
    n.name = std::string(prefix) + std::to_string(id);
    return append(std::move(n));
  }
};

inline ASTContext *&current_ast() noexcept
{
  thread_local ASTContext *ctx = nullptr;
  return ctx;
}

class ASTScope {
  ASTContext *previous_;

public:
  explicit ASTScope(ASTContext &ctx) : previous_(current_ast()) { current_ast() = &ctx; }
  ~ASTScope() { current_ast() = previous_; }
  ASTScope(const ASTScope &) = delete;
  ASTScope &operator=(const ASTScope &) = delete;
};

inline ASTContext &ast()
{
  ASTContext *ctx = current_ast();
  if (ctx == nullptr) { throw std::runtime_error("vlk AST used outside of a tracing scope"); }
  return *ctx;
}

} // namespace vlk
