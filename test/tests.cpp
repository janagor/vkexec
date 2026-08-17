#include <catch2/catch_test_macros.hpp>

#include <vkexec/detail/ast.hpp>
#include <vkexec/detail/glsl_emit.hpp>
#include <vkexec/detail/types.hpp>

#include <string>

TEST_CASE("vkexec AST records arithmetic", "[vkexec]")
{
  vkexec::ASTContext ctx;
  const vkexec::ASTScope scope(ctx);

  vkexec::Float const lhs = vkexec::Float::constant(1.0);
  vkexec::Float const rhs = vkexec::Float::constant(2.0);
  vkexec::Float const result = lhs + (rhs * vkexec::Float::constant(3.0));
  REQUIRE(result.id >= 0);
  REQUIRE(ctx.nodes.size() >= 4);

  std::string const glsl = vkexec::detail::emit_glsl(ctx, 64);
  REQUIRE(glsl.contains("#version 450"));
  REQUIRE(glsl.contains("void main()"));
}
