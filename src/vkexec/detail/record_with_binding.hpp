#ifndef VKEXEC_DETAIL_RECORD_WITH_BINDING_HPP
#define VKEXEC_DETAIL_RECORD_WITH_BINDING_HPP

#include <vkexec/detail/descriptor_backend.hpp>

namespace vkexec::detail {

template<descriptor_backend Backend, class Bind>
[[nodiscard]] auto bind_and_push(context const *ctx,
  VkCommandBuffer cmd,
  VkPipelineBindPoint bind_point,
  Bind const &bind,
  std::span<std::byte const> push) -> status
{
  vkCmdBindPipeline(cmd, bind_point, bind.pipeline);
  Backend::bind_resources(cmd, bind_point, bind);
  return Backend::push(ctx, cmd, bind_point, bind, push);
}

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_RECORD_WITH_BINDING_HPP
