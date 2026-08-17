#include <catch2/catch_test_macros.hpp>

#include <vkexec_edsl/trace.hpp>
#include <vkexec_edsl/types.hpp>

namespace edsl = vkexec::edsl;

TEST_CASE("vkexec eDSL records arithmetic", "[vkexec]")
{
  edsl::trace_scope const scope;

  edsl::Float const lhs = edsl::Float::constant(1.0);
  edsl::Float const rhs = edsl::Float::constant(2.0);
  edsl::Float const result = lhs + (rhs * edsl::Float::constant(3.0));
  REQUIRE(result.id >= 0);
  REQUIRE(lhs.id >= 0);
  REQUIRE(rhs.id >= 0);
}
