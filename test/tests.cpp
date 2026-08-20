#include <catch2/catch_test_macros.hpp>

#include <vkexec_edsl/trace.hpp>
#include <vkexec_edsl/types.hpp>

#include <cstddef>
#include <vector>

namespace edsl = vkexec::edsl;

namespace {

constexpr std::size_t k_default_local_size = 64;
constexpr std::size_t k_left_byte_size = 128;
constexpr std::size_t k_left_elem_count = 32;
constexpr std::size_t k_right_byte_size = 256;
constexpr std::size_t k_right_elem_count = 64;
constexpr std::size_t k_updated_byte_size = 512;
constexpr std::size_t k_updated_elem_count = 128;

}// namespace

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

TEST_CASE("trace_scope records default local size", "[vkexec]")
{
  edsl::trace_scope const scope;
  REQUIRE(scope.local_size_x() == k_default_local_size);
}

TEST_CASE("trace_scope records bound storage buffers", "[vkexec]")
{
  edsl::trace_scope const scope;

  char buffer_left{};
  char buffer_right{};
  int binding_left = -1;
  int binding_right = -1;

  static_cast<void>(
    edsl::bind_storage_buffer(&buffer_left, binding_left, "left", k_left_byte_size, "float", k_left_elem_count));
  static_cast<void>(
    edsl::bind_storage_buffer(&buffer_right, binding_right, "right", k_right_byte_size, "int", k_right_elem_count));

  std::vector<edsl::storage_trace> const traces = scope.buffers();
  REQUIRE(traces.size() == 2);
  REQUIRE(traces.at(0).vk_buffer == &buffer_left);
  REQUIRE(traces.at(0).byte_size == k_left_byte_size);
  REQUIRE(traces.at(0).binding == 0);
  REQUIRE(traces.at(1).vk_buffer == &buffer_right);
  REQUIRE(traces.at(1).byte_size == k_right_byte_size);
  REQUIRE(traces.at(1).binding == 1);

  static_cast<void>(
    edsl::bind_storage_buffer(&buffer_left, binding_left, "left", k_updated_byte_size, "float", k_updated_elem_count));
  std::vector<edsl::storage_trace> const updated = scope.buffers();
  REQUIRE(updated.size() == 2);
  REQUIRE(updated.at(0).byte_size == k_updated_byte_size);
  REQUIRE(updated.at(0).binding == 0);
}
