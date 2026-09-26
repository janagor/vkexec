#ifndef VKEXEC_TENSOR_PASS_HPP
#define VKEXEC_TENSOR_PASS_HPP

//! \file
//! Composed upload, compute, and download passes for staging-backed tensors.

#include <vkexec/pass.hpp>
#include <vkexec/tensor_sync.hpp>

#include <stdexec/execution.hpp>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace vkexec {

template<class Compute, typename... T> struct tensor_pass_closure
{
  Compute compute;
  std::tuple<owned::tensor<T> *...> tensors{};
};

namespace detail {

  template<std::size_t Index = 0, class Graph, class Tuple>
  [[nodiscard]] auto append_tensor_uploads(Graph graph, Tuple const &tensors)
  {
    if constexpr (Index == std::tuple_size_v<Tuple>) {
      return graph;
    } else {
      return append_tensor_uploads<Index + 1>(std::move(graph) | sync_to_device(*std::get<Index>(tensors)), tensors);
    }
  }

  template<std::size_t Index = 0, class Graph, class Tuple>
  [[nodiscard]] auto append_tensor_downloads(Graph graph, Tuple const &tensors)
  {
    if constexpr (Index == std::tuple_size_v<Tuple>) {
      return graph;
    } else {
      return append_tensor_downloads<Index + 1>(std::move(graph) | sync_to_host(*std::get<Index>(tensors)), tensors);
    }
  }

  template<class Graph, class Compute, typename... T>
  [[nodiscard]] auto append_tensor_pass(Graph graph, tensor_pass_closure<Compute, T...> closure)
  {
    auto uploads = append_tensor_uploads(std::move(graph), closure.tensors);
    auto computed = std::move(uploads) | std::move(closure.compute);
    return append_tensor_downloads(std::move(computed), closure.tensors);
  }

}// namespace detail

struct tensor_pass_t
{
  template<detail::static_pass_step Compute, typename... T>
  [[nodiscard]] auto operator()(Compute compute, owned::tensor<T> &...values) const
  {
    static_assert(sizeof...(T) > 0, "tensor_pass requires at least one tensor");
    return tensor_pass_closure<Compute, T...>{ .compute = std::move(compute), .tensors = { &values... } };
  }

  template<detail::push_constant_type Params, typename... T>
  [[nodiscard]] auto
    operator()(compute_bind bind, Params const &params, dispatch groups, owned::tensor<T> &...values) const
  { return (*this)(compute_pass(bind, params, groups), values...); }

  template<detail::push_constant_type Params, typename... T>
  [[nodiscard]] auto operator()(handles::compute_pipeline const &pipe,
    VkDescriptorSet set,
    Params const &params,
    std::uint32_t work_count,
    owned::tensor<T> &...values) const
  { return (*this)(compute_pass(pipe, set, params, work_count), values...); }
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr tensor_pass_t tensor_pass{};

template<class Compute, typename... T>
[[nodiscard]] auto operator|(schedule_sender snd, tensor_pass_closure<Compute, T...> closure)
{ return detail::append_tensor_pass(pass_graph_sender<>{ .ctx = snd.ctx, .steps = {} }, std::move(closure)); }

template<class... Steps, class Compute, typename... T>
[[nodiscard]] auto operator|(pass_graph_sender<Steps...> graph, tensor_pass_closure<Compute, T...> closure)
{ return detail::append_tensor_pass(std::move(graph), std::move(closure)); }

template<class Compute, typename... T>
[[nodiscard]] auto operator|(dynamic_pass_graph_sender graph, tensor_pass_closure<Compute, T...> closure)
  -> dynamic_pass_graph_sender
{ return detail::append_tensor_pass(std::move(graph), std::move(closure)); }

template<class Pred, class Closure> struct tensor_pass_sender
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

template<vkexec_predecessor Pred, class Compute, typename... T>
  requires(!std::same_as<std::remove_cvref_t<Pred>, schedule_sender> && !detail::is_pass_graph_sender_v<Pred>)
[[nodiscard]] auto operator|(Pred &&pred, tensor_pass_closure<Compute, T...> closure)
  -> tensor_pass_sender<std::remove_cvref_t<Pred>, tensor_pass_closure<Compute, T...>>
{
  return tensor_pass_sender<std::remove_cvref_t<Pred>, tensor_pass_closure<Compute, T...>>{
    .pred = std::forward<Pred>(pred),
    .closure = std::move(closure),
  };
}

template<class Pred, class Compute, typename... T, class Env>
[[nodiscard]] auto lower_vkexec_sender(ex::set_value_t /*tag*/,
  tensor_pass_sender<Pred, tensor_pass_closure<Compute, T...>> sndr,
  Env const & /*env*/)
{
  scheduler const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr.pred));
  context *const ctx = sched.get_context();
  return ex::let_value(
    std::move(sndr.pred), [ctx, closure = std::move(sndr.closure)](auto &&...) mutable -> decltype(auto) {
      return detail::append_tensor_pass(pass_graph_sender<>{ .ctx = ctx, .steps = {} }, std::move(closure));
    });
}

}// namespace vkexec

#endif// VKEXEC_TENSOR_PASS_HPP
