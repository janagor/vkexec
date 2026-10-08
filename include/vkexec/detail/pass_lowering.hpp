#ifndef VKEXEC_DETAIL_PASS_LOWERING_HPP
#define VKEXEC_DETAIL_PASS_LOWERING_HPP

// Implementation tail included by pass.hpp after the pass sender and step types
// are complete. This header is intentionally not a standalone public include.

#include <cassert>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace vkexec::detail {

template<class Pred, static_pass_step... Steps> struct pass_chain
{
  Pred pred;
  std::tuple<Steps...> steps;
  std::vector<queue_affinity> step_queues;
  std::vector<std::vector<std::size_t>> step_predecessors;
  std::vector<std::size_t> frontier;
  std::size_t base_index{};
  bool dag_mode{};
  queue_affinity current_queue;
};

struct pass_fragment_root
{
  using sender_concept = ex::sender_t;
  using completion_signatures = ex::completion_signatures<ex::set_value_t()>;
  std::vector<std::size_t> frontier;
  std::size_t base_index{};
  queue_affinity current_queue;

  [[nodiscard]] static auto get_env() noexcept -> scheduler_env { return {}; }
};

[[nodiscard]] inline auto graph_frontier(std::size_t count, std::vector<std::vector<std::size_t>> const &predecessors)
  -> std::vector<std::size_t>
{
  if (count == 0) { return {}; }
  if (predecessors.empty()) { return { count - std::size_t{ 1 } }; }

  std::vector<bool> has_successor(count);
  for (auto const &incoming : predecessors) {
    for (auto const index : incoming) { has_successor.at(index) = true; }
  }
  std::vector<std::size_t> frontier;
  for (std::size_t index{}; index < count; ++index) {
    if (!has_successor.at(index)) { frontier.push_back(index); }
  }
  return frontier;
}

template<class Env>
[[nodiscard]] auto collect_pass_chain(pass_fragment_root root, Env const & /*env*/) -> pass_chain<pass_fragment_root>
{
  return { .pred = {},
    .steps = {},
    .step_queues = {},
    .step_predecessors = {},
    .frontier = std::move(root.frontier),
    .base_index = root.base_index,
    .dag_mode = true,
    .current_queue = root.current_queue };
}

template<class Pred, class Env>
  requires(!std::same_as<std::remove_cvref_t<Pred>, pass_fragment_root>)
[[nodiscard]] auto collect_pass_chain(Pred &&pred, Env const & /*env*/) -> pass_chain<std::decay_t<Pred>>
{
  if constexpr (is_pass_graph_sender_v<Pred>) {
    queue_affinity const queue = pred.current_queue;
    auto const count = pass_step_count(pred.steps);
    auto const dag_mode = !pred.step_predecessors.empty();
    auto frontier = graph_frontier(count, pred.step_predecessors);
    return { .pred = std::forward<Pred>(pred),
      .steps = {},
      .step_queues = {},
      .step_predecessors = {},
      .frontier = std::move(frontier),
      .base_index = count,
      .dag_mode = dag_mode,
      .current_queue = queue };
  } else {
    return { .pred = std::forward<Pred>(pred),
      .steps = {},
      .step_queues = {},
      .step_predecessors = {},
      .frontier = {},
      .base_index = 0,
      .dag_mode = false,
      .current_queue = {} };
  }
}

template<class Pred, static_pass_step... Steps, static_pass_step Step>
[[nodiscard]] auto append_lowered_step(pass_chain<Pred, Steps...> chain, Step step) -> pass_chain<Pred, Steps..., Step>
{
  auto const index = chain.base_index + sizeof...(Steps);
  if (chain.dag_mode) { chain.step_predecessors.push_back(chain.frontier); }
  chain.frontier = { index };
  chain.step_queues.push_back(chain.current_queue);
  return std::apply(
    [&chain, &step](Steps &&...old) -> pass_chain<Pred, Steps..., Step> {
      return { .pred = std::move(chain.pred),
        .steps = { std::move(old)..., std::move(step) },
        .step_queues = std::move(chain.step_queues),
        .step_predecessors = std::move(chain.step_predecessors),
        .frontier = std::move(chain.frontier),
        .base_index = chain.base_index,
        .dag_mode = chain.dag_mode,
        .current_queue = chain.current_queue };
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
  requires(std::same_as<Tag, on_queue_t> || std::same_as<Tag, when_all_t>
            || requires(Tag tag, Data &&data, Env const &env) { lower_vkexec_pass_step(tag, std::move(data), env); })
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

template<std::size_t Index = 0, class Chain, class Branches, class Env>
[[nodiscard]] auto
  collect_pass_branches(Chain chain, Branches branches, std::vector<std::size_t> const &incoming, Env const &env);

template<class Tag, class Data, class Child, class Env>
  requires(std::same_as<Tag, on_queue_t> || std::same_as<Tag, when_all_t>
           || requires(Tag tag, Data &&data, Env const &env) { lower_vkexec_pass_step(tag, std::move(data), env); })
[[nodiscard]] auto collect_pass_chain(sender_expr<Tag, Data, Child> expr, Env const &env)
{
  auto chain = collect_pass_chain(std::move(expr.child), env);
  if constexpr (std::same_as<Tag, on_queue_t>) {
    chain.current_queue = expr.data;
    return chain;
  } else if constexpr (std::same_as<Tag, when_all_t>) {
    if (!chain.dag_mode) {
      for (std::size_t index{}; index < std::tuple_size_v<decltype(chain.steps)>; ++index) {
        std::vector<std::size_t> incoming;
        if (chain.base_index + index != 0) { incoming.push_back(chain.base_index + index - std::size_t{ 1 }); }
        chain.step_predecessors.push_back(std::move(incoming));
      }
      chain.dag_mode = true;
    }
    auto const incoming = chain.frontier;
    chain.frontier.clear();
    return collect_pass_branches(std::move(chain), std::move(expr.data.branches), incoming, env);
  } else {
    auto step = lower_vkexec_pass_step(std::move(expr.tag), std::move(expr.data), env);
    return append_lowered_step(std::move(chain), std::move(step));
  }
}

template<class Pred, static_pass_step... Existing, static_pass_step... Added>
[[nodiscard]] auto merge_pass_branch(pass_chain<Pred, Existing...> chain,
  pass_chain<pass_fragment_root, Added...> branch) -> pass_chain<Pred, Existing..., Added...>
{
  static_assert(sizeof...(Added) != 0, "when_all branches must contain at least one pass");
  chain.step_queues.insert(chain.step_queues.end(), branch.step_queues.begin(), branch.step_queues.end());
  chain.step_predecessors.insert(chain.step_predecessors.end(),
    std::make_move_iterator(branch.step_predecessors.begin()),
    std::make_move_iterator(branch.step_predecessors.end()));
  chain.frontier.insert(chain.frontier.end(), branch.frontier.begin(), branch.frontier.end());
  return { .pred = std::move(chain.pred),
    .steps = std::tuple_cat(std::move(chain.steps), std::move(branch.steps)),
    .step_queues = std::move(chain.step_queues),
    .step_predecessors = std::move(chain.step_predecessors),
    .frontier = std::move(chain.frontier),
    .base_index = chain.base_index,
    .dag_mode = true,
    .current_queue = chain.current_queue };
}

template<std::size_t Index, class Chain, class Branches, class Env>
[[nodiscard]] auto
  collect_pass_branches(Chain chain, Branches branches, std::vector<std::size_t> const &incoming, Env const &env)
{
  if constexpr (Index == std::tuple_size_v<Branches>) {
    return chain;
  } else {
    auto root = pass_fragment_root{ .frontier = incoming,
      .base_index = chain.base_index + std::tuple_size_v<decltype(chain.steps)>,
      .current_queue = chain.current_queue };
    auto expression = std::move(std::get<Index>(branches))(std::move(root));
    auto normalized = normalize_vkexec_expression(std::move(expression), env);
    auto branch = collect_pass_chain(std::move(normalized), env);
    auto merged = merge_pass_branch(std::move(chain), std::move(branch));
    return collect_pass_branches<Index + std::size_t{ 1 }>(std::move(merged), std::move(branches), incoming, env);
  }
}

template<static_pass_step... Steps> struct materialize_pass_graph_fn
{
  context_handle state;
  std::tuple<Steps...> steps;
  std::vector<queue_affinity> step_queues;
  std::vector<std::vector<std::size_t>> step_predecessors;
  queue_affinity current_queue;

  template<class... Values> [[nodiscard]] auto operator()(Values &&.../*values*/) -> pass_graph_sender<Steps...>
  {
    return { .state = std::move(state),
      .steps = std::move(steps),
      .step_queues = std::move(step_queues),
      .step_predecessors = std::move(step_predecessors),
      .presentation = std::nullopt,
      .current_queue = current_queue };
  }
};

template<vkexec_predecessor Pred, static_pass_step... Steps>
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
[[nodiscard]] auto materialize_pass_chain(pass_chain<Pred, Steps...> &&chain)
  -> decltype(ex::let_value(std::declval<Pred>(), std::declval<materialize_pass_graph_fn<Steps...>>()))
{
  scheduler const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(chain.pred));
  auto state = scheduler_access::state(sched);
  return ex::let_value(std::move(chain.pred),
    materialize_pass_graph_fn<Steps...>{ .state = std::move(state),
      .steps = std::move(chain.steps),
      .step_queues = std::move(chain.step_queues),
      .step_predecessors = std::move(chain.step_predecessors),
      .current_queue = chain.current_queue });
}

template<static_pass_step... Steps>
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
[[nodiscard]] auto materialize_pass_chain(pass_chain<schedule_sender, Steps...> &&chain) -> pass_graph_sender<Steps...>
{
  return { .state = std::move(chain.pred.state),
    .steps = std::move(chain.steps),
    .step_queues = std::move(chain.step_queues),
    .step_predecessors = std::move(chain.step_predecessors),
    .presentation = std::nullopt,
    .current_queue = chain.current_queue };
}

template<static_pass_step... Old, static_pass_step... Steps>
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
[[nodiscard]] auto materialize_pass_chain(pass_chain<pass_graph_sender<Old...>, Steps...> &&chain)
  -> pass_graph_sender<Old..., Steps...>
{
  auto state = std::move(chain.pred.state);
  auto queues = std::move(chain.pred.step_queues);
  auto predecessors = std::move(chain.pred.step_predecessors);
  assert(queues.empty() || queues.size() == sizeof...(Old));
  if (queues.empty()) { queues.resize(sizeof...(Old)); }
  queues.insert(queues.end(), chain.step_queues.begin(), chain.step_queues.end());
  if (chain.dag_mode) {
    if (predecessors.empty()) {
      predecessors.resize(sizeof...(Old));
      for (std::size_t index{ 1 }; index < sizeof...(Old); ++index) {
        predecessors.at(index).push_back(index - std::size_t{ 1 });
      }
    }
    predecessors.insert(predecessors.end(),
      std::make_move_iterator(chain.step_predecessors.begin()),
      std::make_move_iterator(chain.step_predecessors.end()));
  }
  return { .state = std::move(state),
    .steps = std::tuple_cat(std::move(chain.pred.steps), std::move(chain.steps)),
    .step_queues = std::move(queues),
    .step_predecessors = std::move(predecessors),
    .presentation = std::move(chain.pred.presentation),
    .current_queue = chain.current_queue };
}

template<static_pass_step... Steps>
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
[[nodiscard]] auto materialize_pass_chain(pass_chain<dynamic_pass_graph_sender, Steps...> &&chain)
  -> dynamic_pass_graph_sender
{
  assert(chain.pred.step_queues.empty() || chain.pred.step_queues.size() == chain.pred.steps.size());
  if (chain.pred.step_queues.empty()) { chain.pred.step_queues.resize(chain.pred.steps.size()); }
  if constexpr (sizeof...(Steps) != 0) {
    if (chain.dag_mode) {
      if (chain.pred.step_predecessors.empty()) {
        chain.pred.step_predecessors.resize(chain.pred.steps.size());
        for (std::size_t index{ 1 }; index < chain.pred.steps.size(); ++index) {
          chain.pred.step_predecessors.at(index).push_back(index - std::size_t{ 1 });
        }
      }
      chain.pred.step_predecessors.insert(chain.pred.step_predecessors.end(),
        std::make_move_iterator(chain.step_predecessors.begin()),
        std::make_move_iterator(chain.step_predecessors.end()));
    }
    std::apply([&chain](Steps &&...step) -> void { (chain.pred.steps.emplace_back(std::move(step)), ...); },
      std::move(chain.steps));
  }
  chain.pred.step_queues.insert(chain.pred.step_queues.end(), chain.step_queues.begin(), chain.step_queues.end());
  chain.pred.current_queue = chain.current_queue;
  return std::move(chain.pred);
}

template<class Step, class Env>
[[nodiscard]] auto lower_vkexec_pass_step(raw_pass_step_t /*tag*/, raw_pass_step_data<Step> data, Env const & /*env*/)
  -> Step
{ return std::move(data.step); }

template<class Uses, class Record, class Env>
[[nodiscard]] auto lower_vkexec_pass_step(custom_pass_t /*tag*/,
  custom_pass_data<Uses, Record> data,
  Env const & /*env*/) -> custom_pass_step<Uses, Record>
{ return { .resources = std::move(data.resources), .record_fn = std::move(data.record_fn) }; }

template<push_constant_type Push, dispatch_kind Dispatch, class Env>
[[nodiscard]] auto lower_vkexec_pass_step(compute_pass_t /*tag*/,
  compute_pass_data<Push, Dispatch> data,
  Env const & /*env*/) -> compute_pass_step<Push, Dispatch>
{ return { .bind = std::move(data.bind), .push = std::move(data.push), .dispatch_info = data.dispatch_info }; }

template<class Tag, class Data, class Child, class Env>
[[nodiscard]] auto lower_vkexec_sender(ex::set_value_t /*tag*/, sender_expr<Tag, Data, Child> expr, Env const &env)
  -> decltype(materialize_pass_chain(collect_pass_chain(normalize_vkexec_expression(std::move(expr), env), env)))
{
  auto normalized = normalize_vkexec_expression(std::move(expr), env);
  return materialize_pass_chain(collect_pass_chain(std::move(normalized), env));
}

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_PASS_LOWERING_HPP
