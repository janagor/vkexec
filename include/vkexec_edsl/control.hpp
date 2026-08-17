#ifndef VKEXEC_EDSL_CONTROL_HPP
#define VKEXEC_EDSL_CONTROL_HPP

#include <vkexec_edsl/trace.hpp>
#include <vkexec_edsl/types.hpp>

#include <utility>

namespace vkexec::edsl {

template<typename Body> auto if_then(Bool condition, Body &&body) -> void
{
  control_if_begin(condition.id);
  std::forward<Body>(body)();
  control_if_end();
}

template<typename ThenBody, typename ElseBody>
auto if_then_else(Bool condition, ThenBody &&then_body, ElseBody &&else_body) -> void
{
  control_if_begin(condition.id);
  std::forward<ThenBody>(then_body)();
  control_else_begin();
  std::forward<ElseBody>(else_body)();
  control_else_end();
}

template<typename Body> auto for_loop(int start, int end, Body &&body) -> void
{
  int const loop_var = control_for_begin(start, end);
  std::forward<Body>(body)(Int{ loop_var });
  control_for_end(loop_var);
}

}// namespace vkexec::edsl

#endif// VKEXEC_EDSL_CONTROL_HPP
