#ifndef VKEXEC_EDSL_TRACE_ACCESS_HPP
#define VKEXEC_EDSL_TRACE_ACCESS_HPP

#include "ast.hpp"

#include <vkexec_edsl/trace.hpp>

namespace vkexec::edsl::detail {

struct trace_ast_access
{
  [[nodiscard]] static auto get(trace_scope const &scope) -> ASTContext const &;
};

}// namespace vkexec::edsl::detail

#endif// VKEXEC_EDSL_TRACE_ACCESS_HPP
