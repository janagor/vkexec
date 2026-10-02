#ifndef VKEXEC_TENSOR_SYNC_HPP
#define VKEXEC_TENSOR_SYNC_HPP

//! \file
//! Pass-graph pipeables that upload/download staging-backed `tensor<T, B>` storage.

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
#if VKEXEC_HAS_EXCEPTIONS
#include <exception>
#endif
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
namespace detail {

  template<typename T, readback_buffer_resource B> struct tensor_sync_data
  {
    tensor<T, B> *target{ nullptr };
  };

  template<typename T, readback_buffer_resource B> [[nodiscard]] auto make_sync_to_device_step(tensor<T, B> *target)
  {
    return make_callback_pass_step([target](context &ctx, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> status {
      if (target == nullptr || target->size() == 0) {
        return fail(errc::invalid_argument, "sync_to_device requires a non-empty tensor");
      }

      auto const bytes = target->byte_size();
      auto staging_map = target->staging().mapped();
      if (staging_map.size() < bytes) { return fail(errc::out_of_range, "sync_to_device staging map is too small"); }
      std::memcpy(staging_map.data(), target->data(), static_cast<std::size_t>(bytes));
      VKEXEC_TRY(target->staging().flush());

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
      return barrier::transfer_to_compute(ctx, cmd);
    });
  }

  template<typename T, readback_buffer_resource B> [[nodiscard]] auto make_sync_to_host_step(tensor<T, B> *target)
  {
    return make_callback_pass_step(
      [target](context & /*ctx*/, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> status {
        if (target == nullptr || target->size() == 0) {
          return fail(errc::invalid_argument, "sync_to_host requires a non-empty tensor");
        }

        auto const bytes = target->byte_size();

        VkBufferMemoryBarrier device_barrier{};
        device_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        device_barrier.srcAccessMask = static_cast<VkAccessFlags>(VK_ACCESS_SHADER_WRITE_BIT)
                                       | static_cast<VkAccessFlags>(VK_ACCESS_TRANSFER_WRITE_BIT);
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
        auto const src_stage = static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
                               | static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_TRANSFER_BIT);
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
      [target]() -> void {
        if (target == nullptr || target->size() == 0) { return; }
        if (auto invalidated = target->staging().invalidate(); !invalidated) { std::terminate(); }
        auto const bytes = static_cast<std::size_t>(target->byte_size());
        auto staging_map = target->staging().mapped();
        if (staging_map.size() < bytes) { return; }
        std::memcpy(target->data(), staging_map.data(), bytes);
      });
  }

}// namespace detail

struct sync_to_device_t
{
  template<typename T, readback_buffer_resource B>
  [[nodiscard]] auto operator()(tensor<T, B> &values) const
    -> detail::expr_closure<sync_to_device_t, detail::tensor_sync_data<T, B>>
  { return detail::make_expr_closure(*this, detail::tensor_sync_data<T, B>{ .target = &values }); }

  template<vkexec_predecessor Sender, typename T, readback_buffer_resource B>
  [[nodiscard]] auto operator()(Sender &&sender, tensor<T, B> &values) const
    -> decltype(std::forward<Sender>(sender) | (*this)(values))
  { return std::forward<Sender>(sender) | (*this)(values); }
};

struct sync_to_host_t
{
  template<typename T, readback_buffer_resource B>
  [[nodiscard]] auto operator()(tensor<T, B> &values) const
    -> detail::expr_closure<sync_to_host_t, detail::tensor_sync_data<T, B>>
  { return detail::make_expr_closure(*this, detail::tensor_sync_data<T, B>{ .target = &values }); }

  template<vkexec_predecessor Sender, typename T, readback_buffer_resource B>
  [[nodiscard]] auto operator()(Sender &&sender, tensor<T, B> &values) const
    -> decltype(std::forward<Sender>(sender) | (*this)(values))
  { return std::forward<Sender>(sender) | (*this)(values); }
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr sync_to_device_t sync_to_device{};
// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr sync_to_host_t sync_to_host{};

namespace detail {

  template<typename T, readback_buffer_resource B, class Env>
  [[nodiscard]] auto lower_vkexec_pass_step(sync_to_device_t /*tag*/, tensor_sync_data<T, B> data, Env const & /*env*/)
    -> decltype(make_sync_to_device_step(data.target))
  { return make_sync_to_device_step(data.target); }

  template<typename T, readback_buffer_resource B, class Env>
  [[nodiscard]] auto lower_vkexec_pass_step(sync_to_host_t /*tag*/, tensor_sync_data<T, B> data, Env const & /*env*/)
    -> decltype(make_sync_to_host_step(data.target))
  { return make_sync_to_host_step(data.target); }

}// namespace detail

}// namespace vkexec

#endif// VKEXEC_TENSOR_SYNC_HPP
