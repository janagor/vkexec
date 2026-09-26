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

namespace detail {

  template<class Params> struct schema_pass_data
  {
    handles::compute_pipeline const *pipe{ nullptr };
    resource_table table;
    [[no_unique_address]] Params params;
    std::uint32_t work_count{ 0 };
  };

}// namespace detail

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

struct schema_pass_t
{
  template<class... Entries, class Params, class... Resources>
    requires(sizeof...(Entries) == sizeof...(Resources)) && std::is_trivially_copyable_v<Params>
  [[nodiscard]] auto operator()(descriptor_schema<Entries...> schema,
    handles::compute_pipeline const &pipe,
    Params const &params,
    std::uint32_t work_count,
    Resources &&...resources) const -> detail::expr_closure<schema_pass_t, detail::schema_pass_data<Params>>
  {
    return detail::make_expr_closure(*this,
      detail::schema_pass_data<Params>{
        .pipe = &pipe,
        .table = make_resource_table(schema, std::forward<Resources>(resources)...),
        .params = params,
        .work_count = work_count,
      });
  }

  template<vkexec_predecessor Sender, class... Args>
    requires requires(schema_pass_t const &self, Sender &&sender, Args &&...args) {
      std::forward<Sender>(sender) | self(std::forward<Args>(args)...);
    }
  [[nodiscard]] auto operator()(Sender &&sender, Args &&...args) const
    -> decltype(std::forward<Sender>(sender) | (*this)(std::forward<Args>(args)...))
  { return std::forward<Sender>(sender) | (*this)(std::forward<Args>(args)...); }
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr schema_pass_t schema_pass{};

namespace detail {

  template<class Params, class Child, class Env>
  [[nodiscard]] auto lower_vkexec_expression(schema_pass_t /*tag*/,
    schema_pass_data<Params> const &data,
    Child child,
    Env const & /*env*/) -> decltype(std::move(child) | bind_resources(*data.pipe, data.table, data.params)
                                     | compute_pass(bind_compute(*data.pipe), groups_for(*data.pipe, data.work_count)))
  {
    auto bind = bind_resources(*data.pipe, data.table, data.params);
    auto compute = compute_pass(bind_compute(*data.pipe), groups_for(*data.pipe, data.work_count));
    return std::move(child) | std::move(bind) | std::move(compute);
  }

}// namespace detail

}// namespace vkexec

#endif// VKEXEC_SCHEMA_PASS_HPP
