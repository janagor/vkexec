#include <catch2/catch_test_macros.hpp>

#include <vkexec/detail/ast.hpp>
#include <vkexec/detail/glsl_emit.hpp>
#include <vkexec/detail/types.hpp>

#include <string>

TEST_CASE("vlk AST records arithmetic", "[vkexec]")
{
  vlk::ASTContext ctx;
  vlk::ASTScope scope(ctx);

  vlk::Float const lhs = vlk::Float::constant(1.0);
  vlk::Float const rhs = vlk::Float::constant(2.0);
  vlk::Float const result = lhs + (rhs * vlk::Float::constant(3.0));
  REQUIRE(result.id >= 0);
  REQUIRE(ctx.nodes.size() >= 4);

  std::string const glsl = vkexec::detail::emit_glsl(ctx, 64);
  REQUIRE(glsl.find("#version 450") != std::string::npos);
  REQUIRE(glsl.find("void main()") != std::string::npos);
}
