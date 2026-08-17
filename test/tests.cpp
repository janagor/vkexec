#include <catch2/catch_test_macros.hpp>

#include <vkexec_edsl/ast.hpp>
#include <vkexec_edsl/glsl_emit.hpp>
#include <vkexec_edsl/types.hpp>

#include <string>

namespace edsl = vkexec::edsl;

TEST_CASE("vkexec AST records arithmetic", "[vkexec]")
{
  edsl::ASTContext ctx;
  edsl::ASTScope const scope(ctx);

  edsl::Float const lhs = edsl::Float::constant(1.0);
  edsl::Float const rhs = edsl::Float::constant(2.0);
  edsl::Float const result = lhs + (rhs * edsl::Float::constant(3.0));
  REQUIRE(result.id >= 0);
  REQUIRE(ctx.nodes.size() >= 4);

  std::string const glsl = edsl::emit_glsl(ctx, 64);
  REQUIRE(glsl.contains("#version 450"));
  REQUIRE(glsl.contains("void main()"));
}
