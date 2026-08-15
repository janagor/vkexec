#pragma once

#include <vkexec/detail/types.hpp>

#include <utility>

namespace vlk {

template<typename Body>
void if_then(Bool condition, Body &&body)
{
  ast().append(ExprNode::make(OpKind::IfBegin, condition.id));
  std::forward<Body>(body)();
  ast().append(ExprNode::make(OpKind::IfEnd));
}

template<typename ThenBody, typename ElseBody>
void if_then_else(Bool condition, ThenBody &&then_body, ElseBody &&else_body)
{
  ast().append(ExprNode::make(OpKind::IfBegin, condition.id));
  std::forward<ThenBody>(then_body)();
  ast().append(ExprNode::make(OpKind::ElseBegin));
  std::forward<ElseBody>(else_body)();
  ast().append(ExprNode::make(OpKind::ElseEnd));
}

template<typename Body>
void for_loop(int start, int end, Body &&body)
{
  const Int begin = Int::constant(start);
  const Int stop = Int::constant(end);
  const int loop_var = ast().make_temp("i");
  emit_assign(loop_var, begin.id);
  ast().append(ExprNode::make(OpKind::ForBegin, loop_var, stop.id));
  std::forward<Body>(body)(Int{ loop_var });
  ast().append(ExprNode::make(OpKind::ForEnd, loop_var));
}

} // namespace vlk
