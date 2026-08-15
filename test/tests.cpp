#include <catch2/catch_test_macros.hpp>

#include <vkexec/sample_library.hpp>
#include <vkexec/detail/ast.hpp>
#include <vkexec/detail/glsl_emit.hpp>
#include <vkexec/detail/types.hpp>

TEST_CASE("factorial of 0 is 1", "[test]")
{
  REQUIRE(Factorial(0) == 1);
}

TEST_CASE("factorial of 1 is 1", "[test]")
{
  REQUIRE(Factorial(1) == 1);
}

TEST_CASE("factorial of 2 is 2", "[test]")
{
  REQUIRE(Factorial(2) == 2);
}

TEST_CASE("factorial of 3 is 6", "[test]")
{
  REQUIRE(Factorial(3) == 6);
}

TEST_CASE("vlk AST records arithmetic", "[vkexec]")
{
  vlk::ASTContext ctx;
  vlk::ASTScope scope(ctx);

  vlk::Float a = vlk::Float::constant(1.0);
  vlk::Float b = vlk::Float::constant(2.0);
  vlk::Float c = a + b * vlk::Float::constant(3.0);
  REQUIRE(c.id >= 0);
  REQUIRE(ctx.nodes.size() >= 4);

  const std::string glsl = vkexec::detail::emit_glsl(ctx, 64);
  REQUIRE(glsl.find("#version 450") != std::string::npos);
  REQUIRE(glsl.find("void main()") != std::string::npos);
}
