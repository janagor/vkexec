#ifndef VKEXEC_DETAIL_DESCRIPTOR_TABLE_BACKEND_HPP
#define VKEXEC_DETAIL_DESCRIPTOR_TABLE_BACKEND_HPP

#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/result.hpp>

#include <concepts>

namespace vkexec::detail {

//! Lowering environment for descriptor sets, which need no external table storage.
struct empty_table_lower_env
{
};

template<class Backend>
concept descriptor_table_backend = requires(context &ctx,
  handles::compute_pipeline const &pipe,
  resource_table const &table,
  Backend::lower_env const &env,
  Backend::bound_type bound) {
  typename Backend::bound_type;
  typename Backend::lower_env;
  { Backend::lower(ctx, pipe, table, env) } -> std::same_as<result<typename Backend::bound_type>>;
  { Backend::make_bind(pipe, bound) } -> std::same_as<compute_bind>;
  { Backend::release(ctx, pipe, bound) } -> std::same_as<void>;
};

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_DESCRIPTOR_TABLE_BACKEND_HPP
