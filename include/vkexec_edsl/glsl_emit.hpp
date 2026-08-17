#ifndef VKEXEC_EDSL_GLSL_EMIT_HPP
#define VKEXEC_EDSL_GLSL_EMIT_HPP

#include <vkexec_edsl/ast.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace vkexec::edsl {

constexpr std::size_t k_hash_golden_ratio = 0x9e3779b97f4a7c15ULL;
constexpr std::size_t k_hash_shift_left = 6U;
constexpr std::size_t k_hash_shift_right = 2U;
constexpr int k_hash_mix_lhs = 3;
constexpr int k_hash_mix_rhs = 7;
constexpr int k_hash_mix_extra = 9;
constexpr int k_hash_mix_fourth = 13;
constexpr int k_hash_mix_binding = 11;

inline auto hash_combine(std::size_t seed, std::size_t value) -> std::size_t
{
  return seed
         ^ (value + k_hash_golden_ratio + (seed << k_hash_shift_left) + (seed >> k_hash_shift_right));
}

inline auto emit_expr(const ASTContext &ctx,
  int node_id,
  std::unordered_map<int, std::string> &names) -> std::string;

inline auto op_symbol(OpKind kind) -> std::string
{
  switch (kind) {
  case OpKind::Add:
    return "+";
  case OpKind::Sub:
    return "-";
  case OpKind::Mul:
    return "*";
  case OpKind::Div:
    return "/";
  case OpKind::Less:
    return "<";
  case OpKind::LessEqual:
    return "<=";
  case OpKind::Greater:
    return ">";
  case OpKind::GreaterEqual:
    return ">=";
  case OpKind::Equal:
    return "==";
  case OpKind::NotEqual:
    return "!=";
  case OpKind::LogicalAnd:
    return "&&";
  case OpKind::LogicalOr:
    return "||";
  default:
    return "?";
  }
}

inline auto emit_expr(const ASTContext &ctx,
  int node_id,
  std::unordered_map<int, std::string> &names) -> std::string
{
  if (node_id < 0) { return "0"; }
  if (auto found = names.find(node_id); found != names.end()) { return found->second; }

  const ExprNode &node = ctx.nodes.at(static_cast<std::size_t>(node_id));
  std::string out;

  switch (node.kind) {
  case OpKind::ConstFloat:
    out = std::format("{:.9f}", node.const_f);
    break;
  case OpKind::ConstInt:
    out = std::to_string(node.const_i);
    break;
  case OpKind::ConstBool:
    out = (node.const_i != 0) ? "true" : "false";
    break;
  case OpKind::Var:
  case OpKind::ParamIndex:
  case OpKind::VertexIndex:
    out = node.name.empty() ? std::format("v{}", node_id) : node.name;
    break;
  case OpKind::PushField:
    out = "pc." + node.name;
    break;
  case OpKind::InputVarying:
    out = node.name;
    break;
  case OpKind::Load: {
    const auto &buf = ctx.buffers.at(static_cast<std::size_t>(node.binding));
    out = std::format("{}[{}]", buf.name, emit_expr(ctx, node.lhs, names));
    break;
  }
  case OpKind::Neg:
    out = "-(" + emit_expr(ctx, node.lhs, names) + ")";
    break;
  case OpKind::LogicalNot:
    out = "!(" + emit_expr(ctx, node.lhs, names) + ")";
    break;
  case OpKind::Sin:
    out = "sin(" + emit_expr(ctx, node.lhs, names) + ")";
    break;
  case OpKind::Cos:
    out = "cos(" + emit_expr(ctx, node.lhs, names) + ")";
    break;
  case OpKind::Sqrt:
    out = "sqrt(" + emit_expr(ctx, node.lhs, names) + ")";
    break;
  case OpKind::Abs:
    out = "abs(" + emit_expr(ctx, node.lhs, names) + ")";
    break;
  case OpKind::Floor:
    out = "floor(" + emit_expr(ctx, node.lhs, names) + ")";
    break;
  case OpKind::Ceil:
    out = "ceil(" + emit_expr(ctx, node.lhs, names) + ")";
    break;
  case OpKind::Min:
    out = "min(" + emit_expr(ctx, node.lhs, names) + ", " + emit_expr(ctx, node.rhs, names) + ")";
    break;
  case OpKind::Max:
    out = "max(" + emit_expr(ctx, node.lhs, names) + ", " + emit_expr(ctx, node.rhs, names) + ")";
    break;
  case OpKind::Vec2:
    out = "vec2(" + emit_expr(ctx, node.lhs, names) + ", " + emit_expr(ctx, node.rhs, names) + ")";
    break;
  case OpKind::Vec3:
    out = "vec3(" + emit_expr(ctx, node.lhs, names) + ", " + emit_expr(ctx, node.rhs, names) + ", "
          + emit_expr(ctx, node.extra, names) + ")";
    break;
  case OpKind::Vec4:
    out = "vec4(" + emit_expr(ctx, node.lhs, names) + ", " + emit_expr(ctx, node.rhs, names) + ", "
          + emit_expr(ctx, node.extra, names) + ", " + emit_expr(ctx, node.fourth, names) + ")";
    break;
  case OpKind::Vec4From2:
    if (node.name == "vec3") {
      out = "vec4(" + emit_expr(ctx, node.lhs, names) + ", " + emit_expr(ctx, node.rhs, names) + ")";
    } else {
      out = "vec4(" + emit_expr(ctx, node.lhs, names) + ", " + emit_expr(ctx, node.rhs, names) + ", "
            + emit_expr(ctx, node.extra, names) + ")";
    }
    break;
  case OpKind::Select:
    out = std::format("({} ? {} : {})",
      emit_expr(ctx, node.lhs, names),
      emit_expr(ctx, node.rhs, names),
      emit_expr(ctx, node.extra, names));
    break;
  case OpKind::Add:
  case OpKind::Sub:
  case OpKind::Mul:
  case OpKind::Div:
  case OpKind::Less:
  case OpKind::LessEqual:
  case OpKind::Greater:
  case OpKind::GreaterEqual:
  case OpKind::Equal:
  case OpKind::NotEqual:
  case OpKind::LogicalAnd:
  case OpKind::LogicalOr:
    out = "(" + emit_expr(ctx, node.lhs, names) + " " + op_symbol(node.kind) + " "
          + emit_expr(ctx, node.rhs, names) + ")";
    break;
  default:
    out = "0 /*unsupported*/";
    break;
  }

  names.emplace(node_id, out);
  return out;
}

inline auto hash_ast(const ASTContext &ctx) -> std::size_t
{
  std::size_t hash = ctx.nodes.size();
  hash = std::ranges::fold_left(ctx.nodes, hash, [](std::size_t seed, const ExprNode &node) -> std::size_t {
    seed = hash_combine(seed, static_cast<std::size_t>(node.kind));
    seed = hash_combine(seed, static_cast<std::size_t>(node.type));
    seed = hash_combine(seed, static_cast<std::size_t>(node.lhs) + static_cast<std::size_t>(k_hash_mix_lhs));
    seed = hash_combine(seed, static_cast<std::size_t>(node.rhs) + static_cast<std::size_t>(k_hash_mix_rhs));
    seed = hash_combine(seed, static_cast<std::size_t>(node.extra) + static_cast<std::size_t>(k_hash_mix_extra));
    seed = hash_combine(seed, static_cast<std::size_t>(node.fourth) + static_cast<std::size_t>(k_hash_mix_fourth));
    seed = hash_combine(seed, static_cast<std::size_t>(node.binding) + static_cast<std::size_t>(k_hash_mix_binding));
    return std::ranges::fold_left(node.name, seed, [](std::size_t current, char character) -> std::size_t {
      return hash_combine(current, static_cast<std::size_t>(static_cast<unsigned char>(character)));
    });
  });
  hash = std::ranges::fold_left(ctx.buffers, hash, [](std::size_t seed, const BufferBinding &buffer) -> std::size_t {
    seed = hash_combine(seed, static_cast<std::size_t>(buffer.binding));
    return std::ranges::fold_left(buffer.elem_glsl_type, seed, [](std::size_t current, char character) -> std::size_t {
      return hash_combine(current, static_cast<std::size_t>(static_cast<unsigned char>(character)));
    });
  });
  hash = hash_combine(hash, ctx.push_bytes);
  return hash;
}

inline auto emit_body_statements(std::ostringstream &stream,
  const ASTContext &ctx,
  std::unordered_map<int, std::string> &names) -> void
{
  for (std::size_t index = 0; index < ctx.nodes.size(); ++index) {
    const auto &node = ctx.nodes.at(index);
    if (node.kind == OpKind::Var) {
      const std::string name = node.name.empty() ? std::format("v{}", index) : node.name;
      names[static_cast<int>(index)] = name;
      stream << "  " << glsl_type_name(node.type) << " " << name << ";\n";
    }
  }

  for (std::size_t index = 0; index < ctx.nodes.size(); ++index) {
    const auto &node = ctx.nodes.at(index);
    switch (node.kind) {
    case OpKind::Assign: {
      stream << "  " << emit_expr(ctx, node.lhs, names) << " = " << emit_expr(ctx, node.rhs, names) << ";\n";
      break;
    }
    case OpKind::Store: {
      const auto &buf = ctx.buffers.at(static_cast<std::size_t>(node.binding));
      stream << "  " << buf.name << "[" << emit_expr(ctx, node.lhs, names)
             << "] = " << emit_expr(ctx, node.rhs, names) << ";\n";
      break;
    }
    case OpKind::OutputVarying:
      stream << "  " << node.name << " = " << emit_expr(ctx, node.rhs, names) << ";\n";
      break;
    case OpKind::IfBegin:
      stream << "  if (" << emit_expr(ctx, node.lhs, names) << ") {\n";
      break;
    case OpKind::ElseBegin:
      stream << "  } else {\n";
      break;
    case OpKind::IfEnd:
    case OpKind::ElseEnd:
      stream << "  }\n";
      break;
    case OpKind::ForBegin:
      stream << "  for (; " << emit_expr(ctx, node.lhs, names) << " < " << emit_expr(ctx, node.rhs, names)
             << "; ++" << emit_expr(ctx, node.lhs, names) << ") {\n";
      break;
    case OpKind::ForEnd:
      stream << "  }\n";
      break;
    default:
      break;
    }
  }
}

inline auto emit_glsl(const ASTContext &ctx, std::uint32_t work_count) -> std::string
{
  std::ostringstream stream;
  stream << "#version 450\n";
  stream << "layout(local_size_x = " << ctx.local_size_x << ") in;\n\n";

  for (const auto &buf : ctx.buffers) {
    stream << "layout(set = 0, binding = " << buf.binding << ") buffer Buf" << buf.binding << " {\n";
    stream << "  " << buf.elem_glsl_type << " data[];\n";
    stream << "} " << buf.name << "_block;\n";
    stream << "#define " << buf.name << " " << buf.name << "_block.data\n\n";
  }

  if (!ctx.push_block_glsl.empty()) { stream << ctx.push_block_glsl << "\n"; }

  stream << "void main() {\n";
  stream << "  uint idx = gl_GlobalInvocationID.x;\n";
  stream << "  if (idx >= " << work_count << "u) return;\n";

  std::unordered_map<int, std::string> names;
  for (std::size_t index = 0; index < ctx.nodes.size(); ++index) {
    if (ctx.nodes.at(index).kind == OpKind::ParamIndex) {
      names.emplace(static_cast<int>(index), "int(idx)");
    }
  }

  emit_body_statements(stream, ctx, names);
  stream << "}\n";
  return stream.str();
}

inline auto emit_vertex_glsl(const ASTContext &ctx) -> std::string
{
  std::ostringstream stream;
  stream << "#version 450\n";

  for (const auto &buf : ctx.buffers) {
    stream << "layout(set = 0, binding = " << buf.binding << ") readonly buffer Buf" << buf.binding << " {\n";
    stream << "  " << buf.elem_glsl_type << " data[];\n";
    stream << "} " << buf.name << "_block;\n";
    stream << "#define " << buf.name << " " << buf.name << "_block.data\n\n";
  }

  std::unordered_set<std::string> outs;
  for (const auto &node : ctx.nodes) {
    if (node.kind == OpKind::OutputVarying && node.name != "gl_Position" && node.name != "gl_PointSize"
        && outs.insert(node.name).second) {
      stream << "layout(location = " << node.const_i << ") out " << glsl_type_name(node.type) << " "
             << node.name << ";\n";
    }
  }
  if (!ctx.push_block_glsl.empty()) { stream << ctx.push_block_glsl << "\n"; }

  stream << "void main() {\n";
  std::unordered_map<int, std::string> names;
  for (std::size_t index = 0; index < ctx.nodes.size(); ++index) {
    if (ctx.nodes.at(index).kind == OpKind::VertexIndex) {
      names.emplace(static_cast<int>(index), "int(gl_VertexIndex)");
    }
  }
  emit_body_statements(stream, ctx, names);
  stream << "}\n";
  return stream.str();
}

inline auto emit_fragment_glsl(const ASTContext &ctx) -> std::string
{
  std::ostringstream stream;
  stream << "#version 450\n";

  std::unordered_set<std::string> seen_in;
  std::unordered_set<std::string> seen_out;
  for (const auto &node : ctx.nodes) {
    if (node.kind == OpKind::InputVarying && seen_in.insert(node.name).second) {
      stream << "layout(location = " << node.const_i << ") in " << glsl_type_name(node.type) << " "
             << node.name << ";\n";
    }
    if (node.kind == OpKind::OutputVarying && seen_out.insert(node.name).second) {
      stream << "layout(location = " << node.const_i << ") out " << glsl_type_name(node.type) << " "
             << node.name << ";\n";
    }
  }
  if (!ctx.push_block_glsl.empty()) { stream << ctx.push_block_glsl << "\n"; }

  stream << "void main() {\n";
  std::unordered_map<int, std::string> names;
  emit_body_statements(stream, ctx, names);
  stream << "}\n";
  return stream.str();
}

} // namespace vkexec::edsl

#endif // VKEXEC_EDSL_GLSL_EMIT_HPP
