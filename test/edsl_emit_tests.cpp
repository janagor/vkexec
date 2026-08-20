#include <catch2/catch_test_macros.hpp>

#include <vkexec_edsl/ast.hpp>
#include <vkexec_edsl/glsl_emit.hpp>
#include <vkexec_edsl/trace.hpp>
#include <vkexec_edsl/trace_access.hpp>
#include <vkexec_edsl/types.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace edsl = vkexec::edsl;

namespace {

constexpr std::size_t k_buffer_byte_size = 64;
constexpr std::size_t k_buffer_elem_count = 16;
constexpr std::uint32_t k_work_count = 32;

}// namespace

TEST_CASE("op_symbol maps binary operators", "[vkexec][edsl]")
{
  REQUIRE(edsl::op_symbol(edsl::OpKind::Add) == "+");
  REQUIRE(edsl::op_symbol(edsl::OpKind::Sub) == "-");
  REQUIRE(edsl::op_symbol(edsl::OpKind::Mul) == "*");
  REQUIRE(edsl::op_symbol(edsl::OpKind::Div) == "/");
  REQUIRE(edsl::op_symbol(edsl::OpKind::Less) == "<");
  REQUIRE(edsl::op_symbol(edsl::OpKind::LessEqual) == "<=");
  REQUIRE(edsl::op_symbol(edsl::OpKind::Greater) == ">");
  REQUIRE(edsl::op_symbol(edsl::OpKind::GreaterEqual) == ">=");
  REQUIRE(edsl::op_symbol(edsl::OpKind::Equal) == "==");
  REQUIRE(edsl::op_symbol(edsl::OpKind::NotEqual) == "!=");
  REQUIRE(edsl::op_symbol(edsl::OpKind::LogicalAnd) == "&&");
  REQUIRE(edsl::op_symbol(edsl::OpKind::LogicalOr) == "||");
  REQUIRE(edsl::op_symbol(edsl::OpKind::ConstFloat) == "?");
}

TEST_CASE("glsl_type_name maps value types", "[vkexec][edsl]")
{
  REQUIRE(std::string_view(edsl::glsl_type_name(edsl::ValueType::Float)) == "float");
  REQUIRE(std::string_view(edsl::glsl_type_name(edsl::ValueType::Int)) == "int");
  REQUIRE(std::string_view(edsl::glsl_type_name(edsl::ValueType::Bool)) == "bool");
  REQUIRE(std::string_view(edsl::glsl_type_name(edsl::ValueType::Vec2)) == "vec2");
  REQUIRE(std::string_view(edsl::glsl_type_name(edsl::ValueType::Vec3)) == "vec3");
  REQUIRE(std::string_view(edsl::glsl_type_name(edsl::ValueType::Vec4)) == "vec4");
}

TEST_CASE("hash_ast is stable and sensitive to AST changes", "[vkexec][edsl]")
{
  edsl::ASTContext ctx;
  ctx.nodes.push_back(edsl::ExprNode::make(edsl::OpKind::ConstFloat));
  ctx.nodes.back().const_f = 1.0;

  std::size_t const initial_hash = edsl::hash_ast(ctx);
  REQUIRE(initial_hash == edsl::hash_ast(ctx));

  ctx.nodes.back().name = "changed";
  REQUIRE(edsl::hash_ast(ctx) != initial_hash);
}

TEST_CASE("emit_glsl includes compute layout from traced kernel", "[vkexec][edsl]")
{
  edsl::trace_scope const scope;

  char storage{};
  int binding = -1;
  static_cast<void>(
    edsl::bind_storage_buffer(&storage, binding, "data", k_buffer_byte_size, "float", k_buffer_elem_count));

  edsl::Int const idx = edsl::Int::param_index();
  edsl::Float const value = edsl::Float::constant(2.0);
  edsl::store_buffer(binding, idx.id, value.id);

  std::string const glsl = edsl::emit_glsl(edsl::detail::trace_ast_access::get(scope), k_work_count);

  REQUIRE(glsl.contains("#version 450"));
  REQUIRE(glsl.contains("layout(local_size_x = 64) in;"));
  REQUIRE(glsl.contains("layout(set = 0, binding = 0) buffer Buf0"));
  REQUIRE(glsl.contains("#define data data_block.data"));
  REQUIRE(glsl.contains("if (idx >= 32u) return;"));
  REQUIRE(glsl.contains("data[int(idx)] = 2.000000000;"));
}
