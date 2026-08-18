#include <catch2/catch_test_macros.hpp>

#include <vkexec_edsl/push_constant.hpp>
#include <vkexec_edsl/trace.hpp>
#include <vkexec_edsl/trace_access.hpp>

#include <boost/describe/class.hpp>

#include <cstdint>
#include <string>

namespace edsl = vkexec::edsl;

namespace {

struct mixed_params
{
  float dt;
  std::uint32_t count;
  int offset;
  bool enabled;
};
// cppcheck-suppress unknownMacro
BOOST_DESCRIBE_STRUCT(mixed_params, (), (dt, count, offset, enabled))

struct sim_params
{
  float dt;
  float damping;
};
// cppcheck-suppress unknownMacro
BOOST_DESCRIBE_STRUCT(sim_params, (), (dt, damping))

}// namespace

TEST_CASE("push_constant glsl_block emits described fields", "[vkexec][push_constant]")
{
  std::string const block = edsl::push_constant<mixed_params>::glsl_block();

  REQUIRE(block.contains("layout(push_constant) uniform PushConstants {"));
  REQUIRE(block.contains("float dt;"));
  REQUIRE(block.contains("uint count;"));
  REQUIRE(block.contains("int offset;"));
  REQUIRE(block.contains("bool enabled;"));
  REQUIRE(block.contains("} pc;"));
}

TEST_CASE("push_constant reports described struct size", "[vkexec][push_constant]")
{
  REQUIRE(edsl::push_constant<mixed_params>::k_byte_size == sizeof(mixed_params));
  REQUIRE(edsl::push_constant<mixed_params>::k_member_count == 4);
}

TEST_CASE("push_constant bind records push block in trace", "[vkexec][push_constant]")
{
  edsl::trace_scope const scope;
  edsl::push_constant<sim_params> const proxy = edsl::push_constant<sim_params>::bind();

  REQUIRE(proxy.get<&sim_params::dt>().id >= 0);
  REQUIRE(proxy.get<&sim_params::damping>().id >= 0);

  auto const &ast = edsl::detail::trace_ast_access::get(scope);
  REQUIRE(ast.push_bytes == sizeof(sim_params));
  REQUIRE(ast.push_block_glsl.contains("float dt;"));
  REQUIRE(ast.push_block_glsl.contains("float damping;"));
}
