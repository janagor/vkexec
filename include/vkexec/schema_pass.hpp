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
    return std::move(bind) | std::move(compute);
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

}// namespace vkexec

#endif// VKEXEC_SCHEMA_PASS_HPP
