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
    Resources &&...resources) const -> bind_resources_closure
  { return bind_resources(pipe, make_resource_table(schema, std::forward<Resources>(resources)...)); }

  template<class... Entries, class Params, class... Resources>
    requires(sizeof...(Entries) == sizeof...(Resources)) && std::is_trivially_copyable_v<Params>
  [[nodiscard]] auto operator()(descriptor_schema<Entries...> schema,
    handles::compute_pipeline const &pipe,
    Params const &params,
    Resources &&...resources) const -> bind_resources_closure
  { return bind_resources(pipe, make_resource_table(schema, std::forward<Resources>(resources)...), params); }
};

//NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr schema_bind_t schema_bind{};

struct schema_pass_closure
{
  bind_resources_closure bind;
  prebuilt_compute_pass_closure compute;
};

namespace detail {

  [[nodiscard]] inline auto append_schema_pass(pass_graph_sender graph, schema_pass_closure closure)
    -> pass_graph_sender
  {
    graph = std::move(graph) | std::move(closure.bind);
    return std::move(graph) | std::move(closure.compute);
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
    Resources &&...resources) const -> schema_pass_closure
  {
    return schema_pass_closure{
      .bind = schema_bind(schema, pipe, params, std::forward<Resources>(resources)...),
      .compute = compute_pass(bind_compute(pipe), groups_for(pipe, work_count)),
    };
  }
};

//NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr schema_pass_t schema_pass{};

[[nodiscard]] inline auto operator|(schedule_sender snd, schema_pass_closure closure) -> pass_graph_sender
{ return detail::append_schema_pass(pass_graph_sender{ .ctx = snd.ctx, .steps = {} }, std::move(closure)); }

[[nodiscard]] inline auto operator|(pass_graph_sender graph, schema_pass_closure closure) -> pass_graph_sender
{ return detail::append_schema_pass(std::move(graph), std::move(closure)); }

template<class Pred> struct schema_pass_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures = pass_graph_sender::completion_signatures;

  Pred pred;
  schema_pass_closure closure;

  [[nodiscard]] auto get_env() const noexcept -> decltype(auto) { return ex::get_env(pred); }
};

template<vkexec_predecessor Pred>
  requires(!std::same_as<std::remove_cvref_t<Pred>, schedule_sender>
           && !std::same_as<std::remove_cvref_t<Pred>, pass_graph_sender>)
[[nodiscard]] auto operator|(Pred &&pred, schema_pass_closure const &closure)
  -> schema_pass_sender<std::remove_cvref_t<Pred>>
{
  return schema_pass_sender<std::remove_cvref_t<Pred>>{
    .pred = std::forward<Pred>(pred),
    .closure = closure,
  };
}

template<class Pred, class Env>
[[nodiscard]] auto lower_vkexec_sender(ex::set_value_t /*tag*/, schema_pass_sender<Pred> sndr, Env const & /*env*/)
{
  scheduler const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr.pred));
  context *const ctx = sched.get_context();
  return ex::let_value(
    std::move(sndr.pred), [ctx, closure = std::move(sndr.closure)](auto &&...) mutable -> pass_graph_sender {
      return detail::append_schema_pass(pass_graph_sender{ .ctx = ctx, .steps = {} }, std::move(closure));
    });
}

}// namespace vkexec

#endif// VKEXEC_SCHEMA_PASS_HPP
