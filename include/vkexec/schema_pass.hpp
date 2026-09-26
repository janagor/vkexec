#ifndef VKEXEC_SCHEMA_PASS_HPP
#define VKEXEC_SCHEMA_PASS_HPP

//! \file
//! Schema-typed classic descriptor binding and compute-pass composition.

#include <vkexec/bind_resources.hpp>
#include <vkexec/descriptor_schema.hpp>
#include <vkexec/pass.hpp>

#include <stdexec/execution.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace vkexec {

struct schema_bind_t
{
  template<class... Entries, class... Resources>
    requires(sizeof...(Entries) == sizeof...(Resources))
  [[nodiscard]] auto operator()(descriptor_schema<Entries...> schema,
    handles::compute_pipeline const &pipe,
    Resources &&...resources) const
  { return bind_resources(pipe, make_resource_table(schema, std::forward<Resources>(resources)...)); }

  template<class... Entries, class Params, class... Resources>
    requires(sizeof...(Entries) == sizeof...(Resources)) && std::is_trivially_copyable_v<Params>
  [[nodiscard]] auto operator()(descriptor_schema<Entries...> schema,
    handles::compute_pipeline const &pipe,
    Params const &params,
    Resources &&...resources) const
  { return bind_resources(pipe, make_resource_table(schema, std::forward<Resources>(resources)...), params); }
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr schema_bind_t schema_bind{};

template<detail::static_pass_step Bind, detail::static_pass_step Compute> struct schema_pass_closure
{
  Bind bind;
  Compute compute;
};

namespace detail {

  template<class... Steps, static_pass_step Bind, static_pass_step Compute>
  [[nodiscard]] auto
    append_schema_pass(pass_graph_sender<Steps...> graph, schema_pass_closure<Bind, Compute> closure)
  {
    auto bound = std::move(graph) | std::move(closure.bind);
    return std::move(bound) | std::move(closure.compute);
  }

}// namespace detail

struct schema_pass_t
{
  template<class... Entries, class Params, class... Resources>
    requires(sizeof...(Entries) == sizeof...(Resources)) && std::is_trivially_copyable_v<Params>
  [[nodiscard]] auto operator()(descriptor_schema<Entries...> schema,
    handles::compute_pipeline const &pipe,
    Params const &params,
    std::uint32_t work_count,
    Resources &&...resources) const
  {
    auto bind = schema_bind(schema, pipe, params, std::forward<Resources>(resources)...);
    auto compute = compute_pass(bind_compute(pipe), groups_for(pipe, work_count));
    return schema_pass_closure<decltype(bind), decltype(compute)>{
      .bind = std::move(bind),
      .compute = std::move(compute),
    };
  }
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr schema_pass_t schema_pass{};

template<detail::static_pass_step Bind, detail::static_pass_step Compute>
[[nodiscard]] auto operator|(schedule_sender snd, schema_pass_closure<Bind, Compute> closure)
{ return detail::append_schema_pass(pass_graph_sender<>{ .ctx = snd.ctx, .steps = {} }, std::move(closure)); }

template<class... Steps, detail::static_pass_step Bind, detail::static_pass_step Compute>
[[nodiscard]] auto operator|(pass_graph_sender<Steps...> graph, schema_pass_closure<Bind, Compute> closure)
{ return detail::append_schema_pass(std::move(graph), std::move(closure)); }

template<detail::static_pass_step Bind, detail::static_pass_step Compute>
[[nodiscard]] auto operator|(dynamic_pass_graph_sender graph, schema_pass_closure<Bind, Compute> closure)
  -> dynamic_pass_graph_sender
{
  graph = std::move(graph) | std::move(closure.bind);
  return std::move(graph) | std::move(closure.compute);
}

template<class Pred, class Closure> struct schema_pass_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures = pass_graph_completion_signatures;

  Pred pred;
  Closure closure;

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env
  {
    scheduler const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(pred));
    return scheduler_env{ .ctx = sched.get_context() };
  }
};

template<vkexec_predecessor Pred, detail::static_pass_step Bind, detail::static_pass_step Compute>
  requires(!std::same_as<std::remove_cvref_t<Pred>, schedule_sender> && !detail::is_pass_graph_sender_v<Pred>)
[[nodiscard]] auto operator|(Pred &&pred, schema_pass_closure<Bind, Compute> closure)
  -> schema_pass_sender<std::remove_cvref_t<Pred>, schema_pass_closure<Bind, Compute>>
{
  return schema_pass_sender<std::remove_cvref_t<Pred>, schema_pass_closure<Bind, Compute>>{
    .pred = std::forward<Pred>(pred),
    .closure = std::move(closure),
  };
}

template<class Pred, class Closure, class Env>
[[nodiscard]] auto
  lower_vkexec_sender(ex::set_value_t /*tag*/, schema_pass_sender<Pred, Closure> sndr, Env const & /*env*/)
{
  scheduler const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr.pred));
  context *const ctx = sched.get_context();
  return ex::let_value(std::move(sndr.pred), [ctx, closure = std::move(sndr.closure)](auto &&...) mutable -> decltype(auto) {
    return detail::append_schema_pass(pass_graph_sender<>{ .ctx = ctx, .steps = {} }, std::move(closure));
  });
}

}// namespace vkexec

#endif// VKEXEC_SCHEMA_PASS_HPP
