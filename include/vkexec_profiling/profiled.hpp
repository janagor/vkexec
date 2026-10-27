#ifndef VKEXEC_PROFILING_PROFILED_HPP
#define VKEXEC_PROFILING_PROFILED_HPP

#include <vkexec/pass.hpp>
#include <vkexec_profiling/detail/profiled_pass_step.hpp>
#include <vkexec_profiling/profiler.hpp>

#include <string>
#include <type_traits>
#include <utility>

namespace vkexec::profiling {

namespace detail {
  struct profiled_t
  {
  };

  template<class Pass> struct profiled_data
  {
    profiler owner;
    std::string name;
    Pass pass;
  };

  template<class Pass, class Env>
  [[nodiscard]] auto lower_vkexec_pass_step(profiled_t /*tag*/, profiled_data<Pass> data, Env const &env) -> auto
  {
    using vkexec::detail::lower_vkexec_pass_step;
    auto inner = lower_vkexec_pass_step(std::move(data.pass.tag), std::move(data.pass.data), env);
    using step_t = decltype(inner);
    static_assert(vkexec::detail::static_pass_step<step_t>, "profiled requires one primitive pass step");
    return profiled_pass_step<step_t>{ std::move(inner), std::move(data.name), std::move(data.owner) };
  }
}// namespace detail

//! Instruments one primitive pass without adding a graph node or dependency.
//! The pass must be recorded outside an active render pass or dynamic rendering scope.
template<class Tag, class Data>
[[nodiscard]] auto profiled(profiler const &owner, std::string const &name, vkexec::detail::expr_closure<Tag, Data> pass)
  -> vkexec::detail::expr_closure<detail::profiled_t,
    detail::profiled_data<vkexec::detail::expr_closure<Tag, Data>>>
{
  using pass_t = vkexec::detail::expr_closure<Tag, Data>;
  return vkexec::detail::make_expr_closure(detail::profiled_t{},
    detail::profiled_data<pass_t>{ .owner = owner, .name = name, .pass = std::move(pass) });
}

}// namespace vkexec::profiling

#endif// VKEXEC_PROFILING_PROFILED_HPP
