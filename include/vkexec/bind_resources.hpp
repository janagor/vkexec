#ifndef VKEXEC_BIND_RESOURCES_HPP
#define VKEXEC_BIND_RESOURCES_HPP

//! \file
//! Pipeable resource-table lowering and descriptor binding.

#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/scheduler.hpp>

#include <cstddef>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

namespace detail {
  struct bind_resources_step_state;

  struct dynamic_bind_resources_data
  {
    handles::compute_pipeline const *pipe{ nullptr };
    resource_table table;
    std::vector<std::byte> push;
  };

  template<push_constant_type Push> struct bind_resources_data
  {
    handles::compute_pipeline const *pipe{ nullptr };
    resource_table table;
    [[no_unique_address]] Push push{};
  };

  [[nodiscard]] auto record_bind_resources_step(context &ctx,
    VkCommandBuffer cmd,
    handles::compute_pipeline const *pipe,
    resource_table const &table,
    std::span<std::byte const> push,
    std::shared_ptr<bind_resources_step_state> &state) -> status;

  auto release_bind_resources_step(std::shared_ptr<bind_resources_step_state> const &state) -> void;
}// namespace detail

//! Descriptor-set resource binding step for a pass graph.
struct runtime_bind_resources_step
{
  handles::compute_pipeline const *pipe{ nullptr };
  resource_table table;
  std::vector<std::byte> push;
  std::shared_ptr<detail::bind_resources_step_state> state;

  auto record(context &ctx, VkCommandBuffer cmd, detail::pass_cleanup &cleanup) -> status;
  auto after_gpu() const -> void;
};

template<detail::push_constant_type Push> struct bind_resources_step
{
  handles::compute_pipeline const *pipe{ nullptr };
  resource_table table;
  [[no_unique_address]] Push push{};
  std::shared_ptr<detail::bind_resources_step_state> state;

  auto record(context &ctx, VkCommandBuffer cmd, detail::pass_cleanup & /*cleanup*/) -> status
  { return detail::record_bind_resources_step(ctx, cmd, pipe, table, detail::push_bytes(push), state); }

  auto after_gpu() const -> void { detail::release_bind_resources_step(state); }
};

//! Builds a descriptor-set-backed resource-table graph step with runtime-sized push data.
[[nodiscard]] auto make_bind_resources_step(handles::compute_pipeline const &pipe,
  resource_table const &table,
  std::span<std::byte const> push) -> runtime_bind_resources_step;

struct bind_resources_t
{
  template<class... Args>
    requires requires(bind_resources_t const &self, Args &&...args) {
      bind_resources_custom(self, std::forward<Args>(args)...);
    }
  [[nodiscard]] auto operator()(Args &&...args) const
    -> decltype(bind_resources_custom(*this, std::forward<Args>(args)...))
  { return bind_resources_custom(*this, std::forward<Args>(args)...); }

  [[nodiscard]] auto operator()(handles::compute_pipeline const &pipe,
    resource_table const &table,
    std::span<std::byte const> push) const
    -> detail::expr_closure<bind_resources_t, detail::dynamic_bind_resources_data>
  {
    return detail::make_expr_closure(*this,
      detail::dynamic_bind_resources_data{
        .pipe = &pipe, .table = table, .push = std::vector<std::byte>{ push.begin(), push.end() } });
  }

  template<std::size_t Extent>
  [[nodiscard]] auto operator()(handles::compute_pipeline const &pipe,
    resource_table const &table,
    std::span<std::byte const, Extent> push) const
    -> detail::expr_closure<bind_resources_t, detail::dynamic_bind_resources_data>
  { return (*this)(pipe, table, std::span<std::byte const>{ push }); }

  template<std::size_t Extent>
  [[nodiscard]] auto operator()(handles::compute_pipeline const &pipe,
    resource_table const &table,
    std::span<std::byte, Extent> push) const
    -> detail::expr_closure<bind_resources_t, detail::dynamic_bind_resources_data>
  { return (*this)(pipe, table, std::span<std::byte const>{ push }); }

  [[nodiscard]] auto operator()(handles::compute_pipeline const &pipe, resource_table const &table) const
    -> detail::expr_closure<bind_resources_t, detail::bind_resources_data<detail::no_push_constants>>
  {
    return detail::make_expr_closure(
      *this, detail::bind_resources_data<detail::no_push_constants>{ .pipe = &pipe, .table = table, .push = {} });
  }

  template<class Params>
    requires std::is_trivially_copyable_v<Params> && (!detail::is_byte_span_v<Params>)
  [[nodiscard]] auto operator()(handles::compute_pipeline const &pipe,
    resource_table const &table,
    Params const &params) const -> detail::expr_closure<bind_resources_t, detail::bind_resources_data<Params>>
  {
    return detail::make_expr_closure(
      *this, detail::bind_resources_data<Params>{ .pipe = &pipe, .table = table, .push = params });
  }

  template<vkexec_predecessor Sender, class... Args>
    requires requires(bind_resources_t const &self, Sender &&sender, Args &&...args) {
      std::forward<Sender>(sender) | self(std::forward<Args>(args)...);
    }
  [[nodiscard]] auto operator()(Sender &&sender, Args &&...args) const
    -> decltype(std::forward<Sender>(sender) | (*this)(std::forward<Args>(args)...))
  { return std::forward<Sender>(sender) | (*this)(std::forward<Args>(args)...); }
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr bind_resources_t bind_resources{};

namespace detail {

  template<class Env>
  [[nodiscard]] auto lower_vkexec_pass_step(bind_resources_t /*tag*/,
    dynamic_bind_resources_data data,
    Env const & /*env*/) -> runtime_bind_resources_step
  {
    return runtime_bind_resources_step{
      .pipe = data.pipe, .table = std::move(data.table), .push = std::move(data.push), .state = {}
    };
  }

  template<push_constant_type Push, class Env>
  [[nodiscard]] auto lower_vkexec_pass_step(bind_resources_t /*tag*/,
    bind_resources_data<Push> data,
    Env const & /*env*/) -> bind_resources_step<Push>
  { return { .pipe = data.pipe, .table = std::move(data.table), .push = std::move(data.push), .state = {} }; }

}// namespace detail

}// namespace vkexec

#endif// VKEXEC_BIND_RESOURCES_HPP
