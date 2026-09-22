#ifndef VKEXEC_TENSOR_PASS_HPP
#define VKEXEC_TENSOR_PASS_HPP

//! \file
//! Composed upload, compute, and download passes for staging-backed tensors.

#include <vkexec/pass.hpp>
#include <vkexec/tensor_sync.hpp>

#include <stdexec/execution.hpp>

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace vkexec {

template<typename... T> struct tensor_pass_closure
{
  prebuilt_compute_pass_closure compute;
  std::tuple<owned::tensor<T> *...> tensors{};
};

namespace detail {

  template<typename... T>
  [[nodiscard]] auto append_tensor_pass(pass_graph_sender graph, tensor_pass_closure<T...> closure) -> pass_graph_sender
  {
    std::apply(
      [&graph](auto *...values) -> void { ((graph = std::move(graph) | sync_to_device(*values)), ...); }, closure.tensors);
    graph = std::move(graph) | std::move(closure.compute);
    std::apply(
      [&graph](auto *...values) -> void { ((graph = std::move(graph) | sync_to_host(*values)), ...); }, closure.tensors);
    return graph;
  }

}// namespace detail

struct tensor_pass_t
{
  template<typename... T>
  [[nodiscard]] auto operator()(prebuilt_compute_pass_closure const &compute, owned::tensor<T> &...values) const
    -> tensor_pass_closure<T...>
  {
    static_assert(sizeof...(T) > 0, "tensor_pass requires at least one tensor");
    return tensor_pass_closure<T...>{ .compute = compute, .tensors = { &values... } };
  }

  template<typename Params, typename... T>
  [[nodiscard]] auto operator()(compute_bind bind, Params const &params, dispatch groups, owned::tensor<T> &...values) const
    -> tensor_pass_closure<T...>
  { return (*this)(compute_pass(bind, params, groups), values...); }

  template<typename Params, typename... T>
  [[nodiscard]] auto operator()(handles::compute_pipeline const &pipe,
    VkDescriptorSet set,
    Params const &params,
    std::uint32_t work_count,
    owned::tensor<T> &...values) const -> tensor_pass_closure<T...>
  { return (*this)(compute_pass(pipe, set, params, work_count), values...); }
};

//NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr tensor_pass_t tensor_pass{};

template<typename... T>
[[nodiscard]] auto operator|(schedule_sender snd, tensor_pass_closure<T...> closure) -> pass_graph_sender
{ return detail::append_tensor_pass(pass_graph_sender{ .ctx = snd.ctx, .steps = {} }, std::move(closure)); }

template<typename... T>
[[nodiscard]] auto operator|(pass_graph_sender graph, tensor_pass_closure<T...> closure) -> pass_graph_sender
{ return detail::append_tensor_pass(std::move(graph), std::move(closure)); }

template<class Pred, class Closure> struct tensor_pass_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures = pass_graph_sender::completion_signatures;

  Pred pred;
  Closure closure;

  [[nodiscard]] auto get_env() const noexcept -> decltype(auto) { return ex::get_env(pred); }
};

template<vkexec_predecessor Pred, typename... T>
  requires(!std::same_as<std::remove_cvref_t<Pred>, schedule_sender>
           && !std::same_as<std::remove_cvref_t<Pred>, pass_graph_sender>)
[[nodiscard]] auto operator|(Pred &&pred, tensor_pass_closure<T...> closure)
  -> tensor_pass_sender<std::remove_cvref_t<Pred>, tensor_pass_closure<T...>>
{
  return tensor_pass_sender<std::remove_cvref_t<Pred>, tensor_pass_closure<T...>>{
    .pred = std::forward<Pred>(pred),
    .closure = std::move(closure),
  };
}

template<class Pred, typename... T, class Env>
[[nodiscard]] auto lower_vkexec_sender(ex::set_value_t /*tag*/,
  tensor_pass_sender<Pred, tensor_pass_closure<T...>> sndr,
  Env const & /*env*/)
{
  scheduler const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr.pred));
  context *const ctx = sched.get_context();
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  return ex::let_value(
    std::move(sndr.pred), [ctx, closure = std::move(sndr.closure)](auto &&...) mutable -> pass_graph_sender {
      return detail::append_tensor_pass(pass_graph_sender{ .ctx = ctx, .steps = {} }, std::move(closure));
    });
}

}// namespace vkexec

#endif// VKEXEC_TENSOR_PASS_HPP
