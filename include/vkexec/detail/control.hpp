#ifndef VKEXEC_DETAIL_CONTROL_HPP
#define VKEXEC_DETAIL_CONTROL_HPP

#include <vkexec/detail/types.hpp>

#include <utility>

namespace vkexec {

template<typename Body>
auto if_then(Bool condition, Body &&body) -> void
{
  ast().append(ExprNode::make(OpKind::IfBegin, condition.id));
  std::forward<Body>(body)();
  ast().append(ExprNode::make(OpKind::IfEnd));
}

template<typename ThenBody, typename ElseBody>
auto if_then_else(Bool condition, ThenBody &&then_body, ElseBody &&else_body) -> void
{
  ast().append(ExprNode::make(OpKind::IfBegin, condition.id));
  std::forward<ThenBody>(then_body)();
  ast().append(ExprNode::make(OpKind::ElseBegin));
  std::forward<ElseBody>(else_body)();
  ast().append(ExprNode::make(OpKind::ElseEnd));
}

template<typename Body>
auto for_loop(int start, int end, Body &&body) -> void
{
  Int const begin = Int::constant(start);
  Int const stop = Int::constant(end);
  int const loop_var = ast().make_temp("i");
  emit_assign(loop_var, begin.id);
  ast().append(ExprNode::make(OpKind::ForBegin, loop_var, stop.id));
  std::forward<Body>(body)(Int{ loop_var });
  ast().append(ExprNode::make(OpKind::ForEnd, loop_var));
}

} // namespace vkexec

#endif // VKEXEC_DETAIL_CONTROL_HPP
