#ifndef VKEXEC_TENSOR_SYNC_HPP
#define VKEXEC_TENSOR_SYNC_HPP

//! \file
//! Pass-graph pipeables that upload/download staging-backed `tensor<T>` storage.

#include <vkexec/barrier.hpp>
#include <vkexec/copy.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/tensor.hpp>

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>

namespace vkexec {

/**
 * Pass-graph closure that copies a tensor host mirror to device storage.
 *
 * Records: host->staging memcpy, staging->device `vkCmdCopyBuffer`, and
 * `barrier::transfer_to_compute`. Pipe onto `schedule()` or a `pass_graph_sender`.
 *
 * @see sync_to_host, tensor
 */
template<typename T> struct sync_to_device_closure
{
  tensor<T> *target{ nullptr };
};

/**
 * Pass-graph closure that copies device storage back into a tensor host mirror.
 *
 * Records device->staging copy + host visibility barriers. Updates the host mirror
 * after the graph's GPU submit completes (`pass_step::after_gpu`).
 *
 * @see sync_to_device, tensor
 */
template<typename T> struct sync_to_host_closure
{
  tensor<T> *target{ nullptr };
};

//! Builds a `sync_to_device` pipeable for `values`.
template<typename T> [[nodiscard]] auto sync_to_device(tensor<T> &values) noexcept -> sync_to_device_closure<T>
{ return sync_to_device_closure<T>{ .target = &values }; }

//! Builds a `sync_to_host` pipeable for `values`.
template<typename T> [[nodiscard]] auto sync_to_host(tensor<T> &values) noexcept -> sync_to_host_closure<T>
{ return sync_to_host_closure<T>{ .target = &values }; }

namespace detail {

  template<typename T> [[nodiscard]] auto make_sync_to_device_step(sync_to_device_closure<T> closure) -> pass_step
  {
    return pass_step{ .record = [target = closure.target](
                                  context & /*ctx*/, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> status {
                       if (target == nullptr || target->size() == 0) {
                         return fail(errc::invalid_argument, "sync_to_device requires a non-empty tensor");
                       }

                       auto const bytes = target->byte_size();
                       auto staging_map = target->staging().mapped();
                       if (staging_map.size() < bytes) {
                         return fail(errc::out_of_range, "sync_to_device staging map is too small");
                       }
                       std::memcpy(staging_map.data(), target->data(), static_cast<std::size_t>(bytes));

                       VkBufferMemoryBarrier staging_barrier{};
                       staging_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                       staging_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
                       staging_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                       staging_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                       staging_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                       staging_barrier.buffer = target->staging().handle();
                       staging_barrier.offset = 0;
                       staging_barrier.size = bytes;

                       VkBufferMemoryBarrier device_barrier{};
                       device_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                       device_barrier.srcAccessMask = 0;
                       device_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                       device_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                       device_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                       device_barrier.buffer = target->device().handle();
                       device_barrier.offset = 0;
                       device_barrier.size = bytes;

                       std::array<VkBufferMemoryBarrier, 2> barriers{ staging_barrier, device_barrier };
                       vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0,
                         nullptr,
                         static_cast<std::uint32_t>(barriers.size()),
                         barriers.data(),
                         0,
                         nullptr);

                       cmd_copy_buffer(cmd, target->staging().handle(), target->device().handle(), bytes);
                       barrier::transfer_to_compute(cmd);
                       return {};
                     },
      .after_gpu = {} };
  }

  template<typename T> [[nodiscard]] auto make_sync_to_host_step(sync_to_host_closure<T> closure) -> pass_step
  {
    tensor<T> *const target = closure.target;
    return pass_step{
      .record = [target](context & /*ctx*/, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> status {
        if (target == nullptr || target->size() == 0) {
          return fail(errc::invalid_argument, "sync_to_host requires a non-empty tensor");
        }

        auto const bytes = target->byte_size();

        VkBufferMemoryBarrier device_barrier{};
        device_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        // NOLINTBEGIN(hicpp-signed-bitwise)
        device_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        // NOLINTEND(hicpp-signed-bitwise)
        device_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        device_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        device_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        device_barrier.buffer = target->device().handle();
        device_barrier.offset = 0;
        device_barrier.size = bytes;

        VkBufferMemoryBarrier staging_barrier{};
        staging_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        staging_barrier.srcAccessMask = 0;
        staging_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        staging_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        staging_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        staging_barrier.buffer = target->staging().handle();
        staging_barrier.offset = 0;
        staging_barrier.size = bytes;

        std::array<VkBufferMemoryBarrier, 2> before{ device_barrier, staging_barrier };
        // NOLINTBEGIN(hicpp-signed-bitwise)
        auto const src_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        // NOLINTEND(hicpp-signed-bitwise)
        vkCmdPipelineBarrier(cmd,
          src_stage,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          0,
          0,
          nullptr,
          static_cast<std::uint32_t>(before.size()),
          before.data(),
          0,
          nullptr);

        cmd_copy_buffer(cmd, target->device().handle(), target->staging().handle(), bytes);

        VkBufferMemoryBarrier host_barrier{};
        host_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        host_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        host_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        host_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        host_barrier.buffer = target->staging().handle();
        host_barrier.offset = 0;
        host_barrier.size = bytes;

        vkCmdPipelineBarrier(
          cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &host_barrier, 0, nullptr);
        return {};
      },
      .after_gpu = [target]() -> void {
        if (target == nullptr || target->size() == 0) { return; }
        auto const bytes = static_cast<std::size_t>(target->byte_size());
        auto staging_map = target->staging().mapped();
        if (staging_map.size() < bytes) { return; }
        std::memcpy(target->data(), staging_map.data(), bytes);
      },
    };
  }

}// namespace detail

template<typename T>
[[nodiscard]] auto operator|(schedule_sender snd, sync_to_device_closure<T> closure) -> pass_graph_sender
{ return pass_graph_sender{ .ctx = snd.ctx, .steps = { detail::make_sync_to_device_step(closure) } }; }

template<typename T>
[[nodiscard]] auto operator|(pass_graph_sender graph, sync_to_device_closure<T> closure) -> pass_graph_sender
{ return detail::append_step(std::move(graph), detail::make_sync_to_device_step(closure)); }

template<typename T>
[[nodiscard]] auto operator|(schedule_sender snd, sync_to_host_closure<T> closure) -> pass_graph_sender
{ return pass_graph_sender{ .ctx = snd.ctx, .steps = { detail::make_sync_to_host_step(closure) } }; }

template<typename T>
[[nodiscard]] auto operator|(pass_graph_sender graph, sync_to_host_closure<T> closure) -> pass_graph_sender
{ return detail::append_step(std::move(graph), detail::make_sync_to_host_step(closure)); }

template<vkexec_predecessor Pred, typename T>
  requires(!std::same_as<std::remove_cvref_t<Pred>, schedule_sender>
           && !std::same_as<std::remove_cvref_t<Pred>, pass_graph_sender>)
[[nodiscard]] auto operator|(Pred &&pred, sync_to_device_closure<T> closure)
  -> pass_adaptor_sender<std::remove_cvref_t<Pred>, sync_to_device_closure<T>>
{
  return pass_adaptor_sender<std::remove_cvref_t<Pred>, sync_to_device_closure<T>>{
    .pred = std::forward<Pred>(pred),
    .closure = closure,
  };
}

template<vkexec_predecessor Pred, typename T>
  requires(!std::same_as<std::remove_cvref_t<Pred>, schedule_sender>
           && !std::same_as<std::remove_cvref_t<Pred>, pass_graph_sender>)
[[nodiscard]] auto operator|(Pred &&pred, sync_to_host_closure<T> closure)
  -> pass_adaptor_sender<std::remove_cvref_t<Pred>, sync_to_host_closure<T>>
{
  return pass_adaptor_sender<std::remove_cvref_t<Pred>, sync_to_host_closure<T>>{
    .pred = std::forward<Pred>(pred),
    .closure = closure,
  };
}

template<class Pred, typename T, class Env>
[[nodiscard]] auto lower_vkexec_sender(ex::set_value_t /*tag*/,
  pass_adaptor_sender<Pred, sync_to_device_closure<T>> sndr,
  Env const & /*env*/)
{
  scheduler const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr.pred));
  // NOLINTNEXTLINE(misc-const-correctness)
  context *const ctx = sched.get_context();
  return ex::let_value(std::move(sndr.pred), [ctx, closure = sndr.closure](auto &&...) mutable -> pass_graph_sender {
    return pass_graph_sender{ .ctx = ctx, .steps = { detail::make_sync_to_device_step(closure) } };
  });
}

template<class Pred, typename T, class Env>
[[nodiscard]] auto lower_vkexec_sender(ex::set_value_t /*tag*/,
  pass_adaptor_sender<Pred, sync_to_host_closure<T>> sndr,
  Env const & /*env*/)
{
  scheduler const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr.pred));
  // NOLINTNEXTLINE(misc-const-correctness)
  context *const ctx = sched.get_context();
  return ex::let_value(std::move(sndr.pred), [ctx, closure = sndr.closure](auto &&...) mutable -> pass_graph_sender {
    return pass_graph_sender{ .ctx = ctx, .steps = { detail::make_sync_to_host_step(closure) } };
  });
}

}// namespace vkexec

#endif// VKEXEC_TENSOR_SYNC_HPP
