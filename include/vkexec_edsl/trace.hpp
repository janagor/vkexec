#ifndef VKEXEC_EDSL_TRACE_HPP
#define VKEXEC_EDSL_TRACE_HPP

#include <vkexec/error.hpp>
#include <vkexec_edsl/types.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vkexec::edsl {
namespace detail {
  struct trace_ast_access;
}

/// Storage buffer observed while tracing an eDSL kernel.
struct storage_trace
{
  void *vk_buffer{ nullptr };
  std::size_t byte_size{ 0 };
  int binding{ -1 };
};

/// RAII tracing session: eDSL operators record into this scope's AST.
class trace_scope
{
public:
  trace_scope();
  ~trace_scope();
  trace_scope(trace_scope const &) = delete;
  auto operator=(trace_scope const &) -> trace_scope & = delete;
  trace_scope(trace_scope &&) = delete;
  auto operator=(trace_scope &&) -> trace_scope & = delete;

  [[nodiscard]] auto buffers() const -> std::vector<storage_trace>;
  [[nodiscard]] auto local_size_x() const -> std::uint32_t;

private:
  friend struct detail::trace_ast_access;
  struct impl;
  std::unique_ptr<impl> impl_;
};

[[nodiscard]] auto compile_vertex_spirv(trace_scope const &scope, std::uint32_t vulkan_api_version = VK_API_VERSION_1_0)
  -> vkexec::result<std::vector<std::uint32_t>>;
[[nodiscard]] auto compile_fragment_spirv(trace_scope const &scope,
  std::uint32_t vulkan_api_version = VK_API_VERSION_1_0) -> vkexec::result<std::vector<std::uint32_t>>;

[[nodiscard]] auto append_push_field(char const *name, std::int64_t offset) -> int;
auto set_push_block(std::string glsl, std::size_t bytes) -> void;

auto control_if_begin(int cond_id) -> void;
auto control_if_end() -> void;
auto control_else_begin() -> void;
auto control_else_end() -> void;
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
[[nodiscard]] auto control_for_begin(int start, int end) -> int;
auto control_for_end(int loop_var) -> void;

[[nodiscard]] auto bind_storage_buffer(void *vk_buffer,
  int &binding,
  char const *name,
  std::size_t byte_size,
  char const *elem_glsl_type,
  std::size_t elem_count) -> int;
[[nodiscard]] auto load_buffer_float(int binding, int index_id) -> Float;
[[nodiscard]] auto load_buffer_int(int binding, int index_id) -> Int;
auto store_buffer(int binding, int index_id, int value_id) -> void;
// NOLINTEND(bugprone-easily-swappable-parameters)

}// namespace vkexec::edsl

#endif// VKEXEC_EDSL_TRACE_HPP
