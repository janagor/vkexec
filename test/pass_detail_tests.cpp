#include <catch2/catch_test_macros.hpp>

#include <vkexec/submit_scope.hpp>
#include <vkexec_edsl/trace.hpp>

#include <array>
#include <span>

namespace edsl = vkexec::edsl;

TEST_CASE("storage_traces_equal compares buffer bindings", "[vkexec][pass]")
{
  using vkexec::detail::storage_traces_equal;

  char buffer_left{};
  char buffer_right{};
  edsl::storage_trace const left_trace{ .vk_buffer = &buffer_left, .byte_size = 64, .binding = 0 };
  edsl::storage_trace const right_trace{ .vk_buffer = &buffer_right, .byte_size = 64, .binding = 0 };
  edsl::storage_trace const left_copy = left_trace;

  REQUIRE(storage_traces_equal(std::span{ &left_trace, 1 }, std::span{ &left_copy, 1 }));
  REQUIRE_FALSE(storage_traces_equal(std::span{ &left_trace, 1 }, std::span{ &right_trace, 1 }));
  REQUIRE(storage_traces_equal(std::span<edsl::storage_trace const>{}, std::span<edsl::storage_trace const>{}));

  edsl::storage_trace const diff_binding{
    .vk_buffer = left_trace.vk_buffer,
    .byte_size = left_trace.byte_size,
    .binding = 1,
  };
  edsl::storage_trace const diff_size_trace{
    .vk_buffer = left_trace.vk_buffer,
    .byte_size = left_trace.byte_size + 1,
    .binding = left_trace.binding,
  };

  REQUIRE_FALSE(storage_traces_equal(std::span{ &left_trace, 1 }, std::span{ &diff_size_trace, 1 }));
  REQUIRE_FALSE(storage_traces_equal(std::span{ &left_trace, 1 }, std::span{ &diff_binding, 1 }));

  std::array const traces{ left_trace, right_trace };
  std::array const traces_reordered{ right_trace, left_trace };
  REQUIRE_FALSE(storage_traces_equal(std::span{ traces }, std::span{ traces_reordered }));
  REQUIRE(storage_traces_equal(std::span{ traces }, std::span{ traces }));
}
