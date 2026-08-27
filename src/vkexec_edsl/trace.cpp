#include <vkexec/error.hpp>
#include <vkexec_edsl/spirv.hpp>
#include <vkexec_edsl/trace.hpp>
#include <vkexec_edsl/types.hpp>

#include "ast.hpp"
#include "glsl_emit.hpp"
#include "trace_access.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vkexec::edsl {

struct trace_scope::impl
{
  ASTContext ast;
  ASTScope scope;
  impl() : scope(ast) {}
};

trace_scope::trace_scope() : impl_(std::make_unique<impl>()) {}
trace_scope::~trace_scope() = default;

auto trace_scope::buffers() const -> std::vector<storage_trace>
{
  std::vector<storage_trace> out;
  out.reserve(impl_->ast.buffers.size());
  std::ranges::transform(impl_->ast.buffers, std::back_inserter(out), [](BufferBinding const &buffer) -> storage_trace {
    return storage_trace{
      .vk_buffer = buffer.vk_buffer,
      .byte_size = buffer.byte_size,
      .binding = buffer.binding,
    };
  });
  return out;
}

auto trace_scope::local_size_x() const -> std::uint32_t { return static_cast<std::uint32_t>(impl_->ast.local_size_x); }

auto detail::trace_ast_access::get(trace_scope const &scope) -> ASTContext const & { return scope.impl_->ast; }

auto compile_vertex_spirv(trace_scope const &scope, std::uint32_t vulkan_api_version)
  -> vkexec::result<std::vector<std::uint32_t>>
{
  std::string const glsl = emit_vertex_glsl(detail::trace_ast_access::get(scope));
  return compile_glsl_to_spirv(glsl, "vkexec.vert", shader_kind::vertex, vulkan_api_version);
}

auto compile_fragment_spirv(trace_scope const &scope, std::uint32_t vulkan_api_version)
  -> vkexec::result<std::vector<std::uint32_t>>
{
  std::string const glsl = emit_fragment_glsl(detail::trace_ast_access::get(scope));
  return compile_glsl_to_spirv(glsl, "vkexec.frag", shader_kind::fragment, vulkan_api_version);
}

auto append_push_field(char const *name, std::int64_t offset) -> int
{
  ExprNode node = ExprNode::make(OpKind::PushField);
  node.name = name;
  node.const_i = offset;
  return ast().append(std::move(node));
}

auto set_push_block(std::string glsl, std::size_t bytes) -> void
{
  ast().push_block_glsl = std::move(glsl);
  ast().push_bytes = bytes;
}

auto control_if_begin(int cond_id) -> void { ast().append(ExprNode::make(OpKind::IfBegin, cond_id)); }
auto control_if_end() -> void { ast().append(ExprNode::make(OpKind::IfEnd)); }
auto control_else_begin() -> void { ast().append(ExprNode::make(OpKind::ElseBegin)); }
auto control_else_end() -> void { ast().append(ExprNode::make(OpKind::ElseEnd)); }

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
auto control_for_begin(int start, int end) -> int
{
  Int const begin = Int::constant(start);
  Int const stop = Int::constant(end);
  int const loop_var = ast().make_temp("i");
  ast().append(ExprNode::make(OpKind::Assign, loop_var, begin.id));
  ast().append(ExprNode::make(OpKind::ForBegin, loop_var, stop.id));
  return loop_var;
}

auto control_for_end(int loop_var) -> void { ast().append(ExprNode::make(OpKind::ForEnd, loop_var)); }

auto bind_storage_buffer(void *vk_buffer,
  int &binding,
  char const *name,
  std::size_t byte_size,
  char const *elem_glsl_type,
  std::size_t elem_count) -> int
{
  ASTContext &ast_ctx = ast();
  if (binding < 0) { binding = ast_ctx.next_binding++; }
  for (BufferBinding &existing : ast_ctx.buffers) {
    // cppcheck-suppress useStlAlgorithm
    if (existing.binding == binding) {
      existing.vk_buffer = vk_buffer;
      existing.byte_size = byte_size;
      existing.elem_count = elem_count;
      return binding;
    }
  }
  BufferBinding binding_info;
  binding_info.name = name;
  binding_info.elem_glsl_type = elem_glsl_type;
  binding_info.binding = binding;
  binding_info.vk_buffer = vk_buffer;
  binding_info.byte_size = byte_size;
  binding_info.elem_count = elem_count;
  ast_ctx.buffers.push_back(std::move(binding_info));
  if (ast_ctx.next_binding <= binding) { ast_ctx.next_binding = binding + 1; }
  return binding;
}

auto load_buffer_float(int binding, int index_id) -> Float
{
  ExprNode node = ExprNode::make(OpKind::Load, index_id);
  node.binding = binding;
  int const load = ast().append(std::move(node));
  int const tmp = ast().make_temp("f");
  ast().append(ExprNode::make(OpKind::Assign, tmp, load));
  return Float{ tmp };
}

auto load_buffer_int(int binding, int index_id) -> Int
{
  ExprNode node = ExprNode::make(OpKind::Load, index_id);
  node.binding = binding;
  int const load = ast().append(std::move(node));
  int const tmp = ast().make_temp("i");
  ast().append(ExprNode::make(OpKind::Assign, tmp, load));
  return Int{ tmp };
}

auto store_buffer(int binding, int index_id, int value_id) -> void
{
  ExprNode node = ExprNode::make(OpKind::Store, index_id, value_id);
  node.binding = binding;
  ast().append(std::move(node));
}
// NOLINTEND(bugprone-easily-swappable-parameters)

}// namespace vkexec::edsl
