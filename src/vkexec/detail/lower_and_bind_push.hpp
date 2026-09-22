#ifndef VKEXEC_DETAIL_LOWER_AND_BIND_PUSH_HPP
#define VKEXEC_DETAIL_LOWER_AND_BIND_PUSH_HPP

#include <vkexec/detail/descriptor_backend.hpp>
#include <vkexec/detail/descriptor_table_backend.hpp>
#include <vkexec/detail/record_with_binding.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <span>

namespace vkexec::detail {

template<class Backend>
  requires descriptor_backend<Backend> && descriptor_table_backend<Backend>
[[nodiscard]] auto lower_and_bind_push(context &ctx,
  VkCommandBuffer cmd,
  VkPipelineBindPoint bind_point,
  handles::compute_pipeline const &pipe,
  resource_table const &table,
  typename Backend::lower_env const &env,
  std::span<std::byte const> push) -> result<typename Backend::bound_type>
{
  auto lowered = Backend::lower(ctx, pipe, table, env);
  if (!lowered) { return fail(lowered); }
  auto bound = expected_take(lowered);
  compute_bind const bind = Backend::make_bind(pipe, bound);
  if (auto recorded = bind_and_push<Backend>(&ctx, cmd, bind_point, bind, push); !recorded) {
    Backend::release(ctx, pipe, bound);
    return fail(recorded);
  }
  return bound;
}

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_LOWER_AND_BIND_PUSH_HPP
