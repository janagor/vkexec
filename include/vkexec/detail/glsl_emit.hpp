#pragma once

#include <vkexec/detail/ast.hpp>

#include <format>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vkexec::detail {

inline std::string emit_expr(const vlk::ASTContext &ctx, int id, std::unordered_map<int, std::string> &names);

inline std::string op_symbol(vlk::OpKind kind)
{
  using vlk::OpKind;
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

inline std::string emit_expr(const vlk::ASTContext &ctx, int id, std::unordered_map<int, std::string> &names)
{
  using vlk::OpKind;
  if (id < 0) { return "0"; }
  if (auto it = names.find(id); it != names.end()) { return it->second; }

  const vlk::ExprNode &n = ctx.nodes.at(static_cast<std::size_t>(id));
  std::string out;

  switch (n.kind) {
  case OpKind::ConstFloat:
    out = std::format("{:.9f}", n.const_f);
    break;
  case OpKind::ConstInt:
    out = std::to_string(n.const_i);
    break;
  case OpKind::ConstBool:
    out = n.const_i ? "true" : "false";
    break;
  case OpKind::Var:
  case OpKind::ParamIndex:
  case OpKind::VertexIndex:
    out = n.name.empty() ? std::format("v{}", id) : n.name;
    break;
  case OpKind::PushField:
    out = "pc." + n.name;
    break;
  case OpKind::InputVarying:
    out = n.name;
    break;
  case OpKind::Load: {
    const auto &buf = ctx.buffers.at(static_cast<std::size_t>(n.binding));
    out = std::format("{}[{}]", buf.name, emit_expr(ctx, n.lhs, names));
    break;
  }
  case OpKind::Neg:
    out = "-(" + emit_expr(ctx, n.lhs, names) + ")";
    break;
  case OpKind::LogicalNot:
    out = "!(" + emit_expr(ctx, n.lhs, names) + ")";
    break;
  case OpKind::Sin:
    out = "sin(" + emit_expr(ctx, n.lhs, names) + ")";
    break;
  case OpKind::Cos:
    out = "cos(" + emit_expr(ctx, n.lhs, names) + ")";
    break;
  case OpKind::Sqrt:
    out = "sqrt(" + emit_expr(ctx, n.lhs, names) + ")";
    break;
  case OpKind::Abs:
    out = "abs(" + emit_expr(ctx, n.lhs, names) + ")";
    break;
  case OpKind::Floor:
    out = "floor(" + emit_expr(ctx, n.lhs, names) + ")";
    break;
  case OpKind::Ceil:
    out = "ceil(" + emit_expr(ctx, n.lhs, names) + ")";
    break;
  case OpKind::Min:
    out = "min(" + emit_expr(ctx, n.lhs, names) + ", " + emit_expr(ctx, n.rhs, names) + ")";
    break;
  case OpKind::Max:
    out = "max(" + emit_expr(ctx, n.lhs, names) + ", " + emit_expr(ctx, n.rhs, names) + ")";
    break;
  case OpKind::Vec2:
    out = "vec2(" + emit_expr(ctx, n.lhs, names) + ", " + emit_expr(ctx, n.rhs, names) + ")";
    break;
  case OpKind::Vec3:
    out = "vec3(" + emit_expr(ctx, n.lhs, names) + ", " + emit_expr(ctx, n.rhs, names) + ", "
          + emit_expr(ctx, n.extra, names) + ")";
    break;
  case OpKind::Vec4:
    out = "vec4(" + emit_expr(ctx, n.lhs, names) + ", " + emit_expr(ctx, n.rhs, names) + ", "
          + emit_expr(ctx, n.extra, names) + ", " + emit_expr(ctx, n.fourth, names) + ")";
    break;
  case OpKind::Vec4From2:
    if (n.name == "vec3") {
      out = "vec4(" + emit_expr(ctx, n.lhs, names) + ", " + emit_expr(ctx, n.rhs, names) + ")";
    } else {
      out = "vec4(" + emit_expr(ctx, n.lhs, names) + ", " + emit_expr(ctx, n.rhs, names) + ", "
            + emit_expr(ctx, n.extra, names) + ")";
    }
    break;
  case OpKind::Select:
    out = std::format("({} ? {} : {})",
      emit_expr(ctx, n.lhs, names),
      emit_expr(ctx, n.rhs, names),
      emit_expr(ctx, n.extra, names));
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
    out = "(" + emit_expr(ctx, n.lhs, names) + " " + op_symbol(n.kind) + " " + emit_expr(ctx, n.rhs, names) + ")";
    break;
  default:
    out = "0 /*unsupported*/";
    break;
  }

  names.emplace(id, out);
  return out;
}

inline std::size_t hash_ast(const vlk::ASTContext &ctx)
{
  std::size_t h = ctx.nodes.size();
  for (const auto &n : ctx.nodes) {
    h ^= (static_cast<std::size_t>(n.kind) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    h ^= (static_cast<std::size_t>(n.type) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    h ^= (static_cast<std::size_t>(n.lhs + 3) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    h ^= (static_cast<std::size_t>(n.rhs + 7) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    h ^= (static_cast<std::size_t>(n.extra + 9) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    h ^= (static_cast<std::size_t>(n.fourth + 13) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    h ^= (static_cast<std::size_t>(n.binding + 11) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    for (char ch : n.name) {
      const auto c = static_cast<unsigned char>(ch);
      h ^= (static_cast<std::size_t>(c) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    }
  }
  for (const auto &b : ctx.buffers) {
    h ^= (static_cast<std::size_t>(b.binding) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    for (char ch : b.elem_glsl_type) {
      const auto c = static_cast<unsigned char>(ch);
      h ^= (static_cast<std::size_t>(c) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    }
  }
  h ^= (ctx.push_bytes + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
  return h;
}

inline void emit_body_statements(std::ostringstream &ss,
  const vlk::ASTContext &ctx,
  std::unordered_map<int, std::string> &names)
{
  using vlk::OpKind;
  for (std::size_t i = 0; i < ctx.nodes.size(); ++i) {
    const auto &n = ctx.nodes[i];
    if (n.kind == OpKind::Var) {
      const std::string nm = n.name.empty() ? std::format("v{}", i) : n.name;
      names[static_cast<int>(i)] = nm;
      ss << "  " << vlk::glsl_type_name(n.type) << " " << nm << ";\n";
    }
  }

  for (std::size_t i = 0; i < ctx.nodes.size(); ++i) {
    const auto &n = ctx.nodes[i];
    switch (n.kind) {
    case OpKind::Assign: {
      ss << "  " << emit_expr(ctx, n.lhs, names) << " = " << emit_expr(ctx, n.rhs, names) << ";\n";
      break;
    }
    case OpKind::Store: {
      const auto &buf = ctx.buffers.at(static_cast<std::size_t>(n.binding));
      ss << "  " << buf.name << "[" << emit_expr(ctx, n.lhs, names) << "] = " << emit_expr(ctx, n.rhs, names)
         << ";\n";
      break;
    }
    case OpKind::OutputVarying:
      ss << "  " << n.name << " = " << emit_expr(ctx, n.rhs, names) << ";\n";
      break;
    case OpKind::IfBegin:
      ss << "  if (" << emit_expr(ctx, n.lhs, names) << ") {\n";
      break;
    case OpKind::ElseBegin:
      ss << "  } else {\n";
      break;
    case OpKind::IfEnd:
    case OpKind::ElseEnd:
      ss << "  }\n";
      break;
    case OpKind::ForBegin:
      ss << "  for (; " << emit_expr(ctx, n.lhs, names) << " < " << emit_expr(ctx, n.rhs, names) << "; ++"
         << emit_expr(ctx, n.lhs, names) << ") {\n";
      break;
    case OpKind::ForEnd:
      ss << "  }\n";
      break;
    default:
      break;
    }
  }
}

inline std::string emit_glsl(const vlk::ASTContext &ctx, std::uint32_t work_count)
{
  using vlk::OpKind;
  std::ostringstream ss;
  ss << "#version 450\n";
  ss << "layout(local_size_x = " << ctx.local_size_x << ") in;\n\n";

  for (const auto &buf : ctx.buffers) {
    ss << "layout(set = 0, binding = " << buf.binding << ") buffer Buf" << buf.binding << " {\n";
    ss << "  " << buf.elem_glsl_type << " data[];\n";
    ss << "} " << buf.name << "_block;\n";
    ss << "#define " << buf.name << " " << buf.name << "_block.data\n\n";
  }

  if (!ctx.push_block_glsl.empty()) { ss << ctx.push_block_glsl << "\n"; }

  ss << "void main() {\n";
  ss << "  uint idx = gl_GlobalInvocationID.x;\n";
  ss << "  if (idx >= " << work_count << "u) return;\n";

  std::unordered_map<int, std::string> names;
  for (std::size_t i = 0; i < ctx.nodes.size(); ++i) {
    if (ctx.nodes[i].kind == OpKind::ParamIndex) {
      names.emplace(static_cast<int>(i), "int(idx)");
    }
  }

  emit_body_statements(ss, ctx, names);
  ss << "}\n";
  return ss.str();
}

inline std::string emit_vertex_glsl(const vlk::ASTContext &ctx)
{
  using vlk::OpKind;
  std::ostringstream ss;
  ss << "#version 450\n";

  for (const auto &buf : ctx.buffers) {
    ss << "layout(set = 0, binding = " << buf.binding << ") readonly buffer Buf" << buf.binding << " {\n";
    ss << "  " << buf.elem_glsl_type << " data[];\n";
    ss << "} " << buf.name << "_block;\n";
    ss << "#define " << buf.name << " " << buf.name << "_block.data\n\n";
  }

  std::unordered_set<std::string> outs;
  for (const auto &n : ctx.nodes) {
    if (n.kind == OpKind::OutputVarying && n.name != "gl_Position" && n.name != "gl_PointSize"
        && outs.insert(n.name).second) {
      ss << "layout(location = " << n.const_i << ") out " << vlk::glsl_type_name(n.type) << " " << n.name << ";\n";
    }
  }
  if (!ctx.push_block_glsl.empty()) { ss << ctx.push_block_glsl << "\n"; }

  ss << "void main() {\n";
  std::unordered_map<int, std::string> names;
  for (std::size_t i = 0; i < ctx.nodes.size(); ++i) {
    if (ctx.nodes[i].kind == OpKind::VertexIndex) {
      names.emplace(static_cast<int>(i), "int(gl_VertexIndex)");
    }
  }
  emit_body_statements(ss, ctx, names);
  ss << "}\n";
  return ss.str();
}

inline std::string emit_fragment_glsl(const vlk::ASTContext &ctx)
{
  using vlk::OpKind;
  std::ostringstream ss;
  ss << "#version 450\n";

  std::unordered_set<std::string> seen_in;
  std::unordered_set<std::string> seen_out;
  for (const auto &n : ctx.nodes) {
    if (n.kind == OpKind::InputVarying && seen_in.insert(n.name).second) {
      ss << "layout(location = " << n.const_i << ") in " << vlk::glsl_type_name(n.type) << " " << n.name << ";\n";
    }
    if (n.kind == OpKind::OutputVarying && seen_out.insert(n.name).second) {
      ss << "layout(location = " << n.const_i << ") out " << vlk::glsl_type_name(n.type) << " " << n.name << ";\n";
    }
  }
  if (!ctx.push_block_glsl.empty()) { ss << ctx.push_block_glsl << "\n"; }

  ss << "void main() {\n";
  std::unordered_map<int, std::string> names;
  emit_body_statements(ss, ctx, names);
  ss << "}\n";
  return ss.str();
}

} // namespace vkexec::detail
