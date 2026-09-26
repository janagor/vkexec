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

namespace detail {

  template<class First, class... Rest> [[nodiscard]] auto compose_adaptors(First &&first, Rest &&...rest)
  { return (std::forward<First>(first) | ... | std::forward<Rest>(rest)); }

}// namespace detail

struct tensor_pass_t
{
  template<detail::sender_adaptor_closure Compute, typename... T>
  [[nodiscard]] auto operator()(Compute compute, owned::tensor<T> &...values) const
  {
    static_assert(sizeof...(T) > 0, "tensor_pass requires at least one tensor");
    auto adaptors = std::tuple_cat(std::tuple{ sync_to_device(values)... },
      std::tuple{ std::move(compute) },
      std::tuple{ sync_to_host(values)... });
    return std::apply(
      []<class... Adaptors>(Adaptors &&...items) -> decltype(auto) {
        return detail::compose_adaptors(std::forward<Adaptors>(items)...);
      },
      std::move(adaptors));
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

  template<vkexec_predecessor Sender, class... Args>
    requires requires(tensor_pass_t const &self, Sender &&sender, Args &&...args) {
      std::forward<Sender>(sender) | self(std::forward<Args>(args)...);
    }
  [[nodiscard]] auto operator()(Sender &&sender, Args &&...args) const
    -> decltype(std::forward<Sender>(sender) | (*this)(std::forward<Args>(args)...))
  { return std::forward<Sender>(sender) | (*this)(std::forward<Args>(args)...); }
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr tensor_pass_t tensor_pass{};

}// namespace vkexec

#endif// VKEXEC_TENSOR_PASS_HPP
