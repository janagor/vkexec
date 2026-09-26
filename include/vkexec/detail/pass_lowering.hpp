#ifndef VKEXEC_DETAIL_PASS_LOWERING_HPP
#define VKEXEC_DETAIL_PASS_LOWERING_HPP

// Implementation tail included by pass.hpp after the pass sender and step types
// are complete. This header is intentionally not a standalone public include.

#include <tuple>
#include <type_traits>
#include <utility>

namespace vkexec::detail {

template<class Pred, static_pass_step... Steps> struct pass_chain
{
  Pred pred;
  std::tuple<Steps...> steps;
};

template<class Pred, class Env>
[[nodiscard]] auto collect_pass_chain(Pred &&pred, Env const & /*env*/) -> pass_chain<std::decay_t<Pred>>
{ return { .pred = std::forward<Pred>(pred), .steps = {} }; }

template<class Pred, static_pass_step... Steps, static_pass_step Step>
[[nodiscard]] auto append_lowered_step(pass_chain<Pred, Steps...> chain, Step step) -> pass_chain<Pred, Steps..., Step>
{
  return std::apply(
    [&chain, &step](Steps &&...old) -> pass_chain<Pred, Steps..., Step> {
      return { .pred = std::move(chain.pred), .steps = { std::move(old)..., std::move(step) } };
    },
    std::move(chain.steps));
}

template<class Sender, class Env>
  requires(!is_sender_expr_v<Sender>)
[[nodiscard]] auto normalize_vkexec_expression(Sender sender, Env const & /*env*/) -> Sender
{ return sender; }

// Expand composite semantic operations before collecting primitive pass steps,
// so a composite cannot become a submission boundary inside a larger chain.
template<class Tag, class Data, class Child, class Env>
  requires requires(Tag tag, Data &&data, Env const &env) { lower_vkexec_pass_step(tag, std::move(data), env); }
           && (!requires(Tag tag, Data &&data, Child &&child, Env const &env) {
                lower_vkexec_expression(tag, std::move(data), std::move(child), env);
              })
[[nodiscard]] auto normalize_vkexec_expression(sender_expr<Tag, Data, Child> expr, Env const &env)
  -> decltype(make_sender_expr(std::move(expr.tag),
    std::move(expr.data),
    normalize_vkexec_expression(std::move(expr.child), env)))
{
  auto child = normalize_vkexec_expression(std::move(expr.child), env);
  return make_sender_expr(std::move(expr.tag), std::move(expr.data), std::move(child));
}

template<class Tag, class Data, class Child, class Env>
using lowered_vkexec_expression_t = decltype(lower_vkexec_expression(std::declval<Tag>(),
  std::declval<Data>(),
  std::declval<Child>(),
  std::declval<Env const &>()));

template<class Tag, class Data, class Child, class Env>
  requires requires { typename lowered_vkexec_expression_t<Tag, Data &&, Child &&, Env>; }
           && std::same_as<std::remove_cvref_t<lowered_vkexec_expression_t<Tag, Data &&, Child &&, Env>>,
             sender_expr<Tag, Data, Child>>
[[nodiscard]] auto normalize_vkexec_expression(sender_expr<Tag, Data, Child> expr, Env const & /*env*/)
  -> sender_expr<Tag, Data, Child>
{
  using lowered_t = lowered_vkexec_expression_t<Tag, Data &&, Child &&, Env>;
  static_assert(!std::same_as<std::remove_cvref_t<lowered_t>, sender_expr<Tag, Data, Child>>,
    "lower_vkexec_expression must make structural progress");
  return expr;
}

template<class Tag, class Data, class Child, class Env>
  requires requires { typename lowered_vkexec_expression_t<Tag, Data &&, Child &&, Env>; }
           && (!std::same_as<std::remove_cvref_t<lowered_vkexec_expression_t<Tag, Data &&, Child &&, Env>>,
             sender_expr<Tag, Data, Child>>)
[[nodiscard]] auto normalize_vkexec_expression(sender_expr<Tag, Data, Child> expr, Env const &env)
  -> decltype(normalize_vkexec_expression(
    lower_vkexec_expression(std::move(expr.tag), std::move(expr.data), std::move(expr.child), env),
    env))
{
  auto lowered = lower_vkexec_expression(std::move(expr.tag), std::move(expr.data), std::move(expr.child), env);
  return normalize_vkexec_expression(std::move(lowered), env);
}

template<class Tag, class Data, class Child, class Env>
  requires requires(Tag tag, Data &&data, Env const &env) { lower_vkexec_pass_step(tag, std::move(data), env); }
[[nodiscard]] auto collect_pass_chain(sender_expr<Tag, Data, Child> expr, Env const &env)
  -> decltype(append_lowered_step(collect_pass_chain(std::move(expr.child), env),
    lower_vkexec_pass_step(std::move(expr.tag), std::move(expr.data), env)))
{
  auto chain = collect_pass_chain(std::move(expr.child), env);
  auto step = lower_vkexec_pass_step(std::move(expr.tag), std::move(expr.data), env);
  return append_lowered_step(std::move(chain), std::move(step));
}

template<static_pass_step... Steps> struct materialize_pass_graph_fn
{
  context *ctx{ nullptr };
  std::tuple<Steps...> steps;

  template<class... Values> [[nodiscard]] auto operator()(Values &&.../*values*/) -> pass_graph_sender<Steps...>
  { return { .ctx = ctx, .steps = std::move(steps) }; }
};

template<vkexec_predecessor Pred, static_pass_step... Steps>
[[nodiscard]] auto materialize_pass_chain(pass_chain<Pred, Steps...> chain)
  -> decltype(ex::let_value(std::move(chain.pred),
    materialize_pass_graph_fn<Steps...>{ .ctx = nullptr, .steps = std::move(chain.steps) }))
{
  scheduler const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(chain.pred));
  // NOLINTNEXTLINE(misc-const-correctness)
  context *const ctx = sched.get_context();
  return ex::let_value(
    std::move(chain.pred), materialize_pass_graph_fn<Steps...>{ .ctx = ctx, .steps = std::move(chain.steps) });
}

template<static_pass_step... Steps>
[[nodiscard]] auto materialize_pass_chain(pass_chain<schedule_sender, Steps...> chain) -> pass_graph_sender<Steps...>
{ return { .ctx = chain.pred.ctx, .steps = std::move(chain.steps) }; }

template<static_pass_step... Old, static_pass_step... Steps>
[[nodiscard]] auto materialize_pass_chain(pass_chain<pass_graph_sender<Old...>, Steps...> chain)
  -> pass_graph_sender<Old..., Steps...>
{
  context *const ctx = chain.pred.ctx;
  return { .ctx = ctx, .steps = std::tuple_cat(std::move(chain.pred.steps), std::move(chain.steps)) };
}

template<static_pass_step... Steps>
[[nodiscard]] auto materialize_pass_chain(pass_chain<dynamic_pass_graph_sender, Steps...> chain)
  -> dynamic_pass_graph_sender
{
  std::apply([&chain](Steps &&...step) -> void { (chain.pred.steps.emplace_back(std::move(step)), ...); },
    std::move(chain.steps));
  return std::move(chain.pred);
}

template<class Step, class Env>
[[nodiscard]] auto lower_vkexec_pass_step(raw_pass_step_t /*tag*/, raw_pass_step_data<Step> data, Env const & /*env*/)
  -> Step
{ return std::move(data.step); }

template<push_constant_type Push, dispatch_kind Dispatch, class Env>
[[nodiscard]] auto lower_vkexec_pass_step(compute_pass_t /*tag*/,
  compute_pass_data<Push, Dispatch> data,
  Env const & /*env*/) -> compute_pass_step<Push, Dispatch>
{ return { .bind = data.bind, .push = std::move(data.push), .dispatch_info = data.dispatch_info }; }

template<class Tag, class Data, class Child, class Env>
[[nodiscard]] auto lower_vkexec_sender(ex::set_value_t /*tag*/, sender_expr<Tag, Data, Child> expr, Env const &env)
  -> decltype(materialize_pass_chain(collect_pass_chain(normalize_vkexec_expression(std::move(expr), env), env)))
{
  auto normalized = normalize_vkexec_expression(std::move(expr), env);
  return materialize_pass_chain(collect_pass_chain(std::move(normalized), env));
}

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_PASS_LOWERING_HPP
