#ifndef VKEXEC_DETAIL_SUBMIT_SCOPE_HPP
#define VKEXEC_DETAIL_SUBMIT_SCOPE_HPP

#include <vkexec/context.hpp>
#include <vkexec/detail/config.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec_edsl/trace.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vkexec {

namespace ex = stdexec;

namespace detail {

  /// Descriptor sets allocated for one recording/submit, freed together on scope exit.
  struct descriptor_cleanup
  {
    struct allocated_set
    {
      VkDescriptorPool pool{ VK_NULL_HANDLE };
      VkDescriptorSet set{ VK_NULL_HANDLE };
    };

    struct pipeline_set_entry
    {
      std::vector<edsl::storage_trace> buffers;
      VkDescriptorSet set{ VK_NULL_HANDLE };
    };

    std::unordered_map<pipeline_resources *, pipeline_set_entry> sets;
    std::vector<allocated_set> allocated;

    auto release(context const &ctx) noexcept -> void
    {
      std::unique_lock const lock = ctx.lock_host();
      for (allocated_set const &item : allocated) { vkFreeDescriptorSets(ctx.device(), item.pool, 1, &item.set); }
      allocated.clear();
      sets.clear();
    }

    auto track(VkDescriptorPool pool, VkDescriptorSet set) -> void
    { allocated.push_back(allocated_set{ .pool = pool, .set = set }); }
  };

  // Compatibility alias used by pass graph recording.
  using pass_cleanup = descriptor_cleanup;

  inline auto storage_traces_equal(std::span<edsl::storage_trace const> lhs, std::span<edsl::storage_trace const> rhs)
    -> bool
  {
    return lhs.size() == rhs.size()
           && std::equal(lhs.begin(),
             lhs.end(),
             rhs.begin(),
             [](edsl::storage_trace const &left, edsl::storage_trace const &right) -> bool {
               return left.vk_buffer == right.vk_buffer && left.byte_size == right.byte_size
                      && left.binding == right.binding;
             });
  }

  inline auto write_storage_descriptors(VkDevice device,
    VkDescriptorSet set,
    std::span<edsl::storage_trace const> buffers) -> void
  {
    if (buffers.empty()) { return; }
    std::vector<VkDescriptorBufferInfo> buf_infos(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    std::size_t index = 0;
    for (edsl::storage_trace const &buffer : buffers) {
      buf_infos.at(index).buffer = static_cast<VkBuffer>(buffer.vk_buffer);
      buf_infos.at(index).offset = 0;
      buf_infos.at(index).range = buffer.byte_size;
      writes.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes.at(index).dstSet = set;
      writes.at(index).dstBinding = static_cast<std::uint32_t>(buffer.binding);
      writes.at(index).descriptorCount = 1;
      writes.at(index).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes.at(index).pBufferInfo = &buf_infos.at(index);
      ++index;
    }
    vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  inline auto write_traced_descriptors(VkDevice device,
    VkDescriptorSet set,
    std::span<edsl::storage_trace const> buffers) -> void
  { write_storage_descriptors(device, set, buffers); }

  inline auto allocate_compute_set(context const &ctx,
    pipeline_resources &pipe,
    std::span<edsl::storage_trace const> buffers) -> VkDescriptorSet
  {
    std::unique_lock const lock = ctx.lock_host();
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = pipe.descriptor_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &pipe.set_layout;
    VkDescriptorSet set{ VK_NULL_HANDLE };
    if (vkAllocateDescriptorSets(ctx.device(), &dsai, &set) != VK_SUCCESS) {
      VKEXEC_THROW(std::runtime_error("vkAllocateDescriptorSets failed"));
    }
    write_storage_descriptors(ctx.device(), set, buffers);
    return set;
  }

  inline auto bind_or_allocate_set(context const &ctx,
    pipeline_resources &pipe,
    std::span<edsl::storage_trace const> buffers,
    descriptor_cleanup &cleanup) -> VkDescriptorSet
  {
    if (auto found = cleanup.sets.find(&pipe); found != cleanup.sets.end()) {
      if (storage_traces_equal(found->second.buffers, buffers)) { return found->second.set; }
    }

    VkDescriptorSet set = allocate_compute_set(ctx, pipe, buffers);
    cleanup.sets.insert_or_assign(
      &pipe, descriptor_cleanup::pipeline_set_entry{ .buffers = { buffers.begin(), buffers.end() }, .set = set });
    cleanup.track(pipe.descriptor_pool, set);
    return set;
  }

  /// Command buffer + descriptor loans for one GPU submit. Exit always frees both.
  struct submit_scope
  {
    context *ctx{ nullptr };
    VkCommandBuffer cmd{ VK_NULL_HANDLE };
    descriptor_cleanup cleanup{};

    submit_scope() = default;
    submit_scope(submit_scope const &) = delete;
    auto operator=(submit_scope const &) -> submit_scope & = delete;

    submit_scope(submit_scope &&other) noexcept
      : ctx(other.ctx), cmd(other.cmd), cleanup(std::move(other.cleanup))
    {
      other.ctx = nullptr;
      other.cmd = VK_NULL_HANDLE;
    }

    auto operator=(submit_scope &&other) noexcept -> submit_scope &
    {
      if (this == &other) { return *this; }
      release();
      ctx = other.ctx;
      cmd = other.cmd;
      cleanup = std::move(other.cleanup);
      other.ctx = nullptr;
      other.cmd = VK_NULL_HANDLE;
      return *this;
    }

    ~submit_scope() { release(); }

    [[nodiscard]] static auto open(context &host) -> submit_scope
    {
      submit_scope scope;
      scope.ctx = &host;
      scope.cmd = host.allocate_command_buffer();
      VkCommandBufferBeginInfo begin{};
      begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      if (vkBeginCommandBuffer(scope.cmd, &begin) != VK_SUCCESS) {
        host.free_command_buffer(scope.cmd);
        scope.cmd = VK_NULL_HANDLE;
        scope.ctx = nullptr;
        VKEXEC_THROW(std::runtime_error("vkBeginCommandBuffer failed"));
      }
      return scope;
    }

    // NOLINTNEXTLINE(readability-make-member-function-const) -- ends Vulkan recording; not logically const
    auto end_recording() -> void
    {
      if (cmd == VK_NULL_HANDLE) { VKEXEC_THROW(std::runtime_error("submit_scope has no command buffer")); }
      if (vkEndCommandBuffer(cmd) != VK_SUCCESS) { VKEXEC_THROW(std::runtime_error("vkEndCommandBuffer failed")); }
    }

    auto track_set(pipeline_resources const &pipe, VkDescriptorSet set) -> void
    { cleanup.track(pipe.descriptor_pool, set); }

    auto release() noexcept -> void
    {
      if (ctx == nullptr) { return; }
      if (cmd != VK_NULL_HANDLE) {
        ctx->free_command_buffer(cmd);
        cmd = VK_NULL_HANDLE;
      }
      cleanup.release(*ctx);
      ctx = nullptr;
    }
  };

  /// Wait for GPU work, destroy semaphore/fence. Safe when handles are null.
  inline auto reclaim_submission_sync(VkDevice device,
    VkQueue fallback_queue,
    VkSemaphore semaphore,
    VkFence fence) noexcept -> void
  {
    if (fence != VK_NULL_HANDLE) {
      (void)vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    } else if (fallback_queue != VK_NULL_HANDLE) {
      (void)vkQueueWaitIdle(fallback_queue);
    }
    if (semaphore != VK_NULL_HANDLE) { vkDestroySemaphore(device, semaphore, nullptr); }
    if (fence != VK_NULL_HANDLE) { vkDestroyFence(device, fence, nullptr); }
  }

  /// Release cmd/descriptors, then deliver exactly one completion signal.
  /// Call only after GPU/host sync objects for this submit have already been reclaimed.
  template<class Receiver>
  auto complete_after_reclaim(Receiver &&receiver, std::exception_ptr error, bool stopped) -> void
  {
    if (error) {
      ex::set_error(std::forward<Receiver>(receiver), std::move(error));
    } else if (stopped) {
      ex::set_stopped(std::forward<Receiver>(receiver));
    } else {
      ex::set_value(std::forward<Receiver>(receiver));
    }
  }

  template<class Receiver>
  auto release_scope_and_complete(submit_scope &scope, Receiver &&receiver, std::exception_ptr error, bool stopped)
    -> void
  {
    scope.release();
    complete_after_reclaim(std::forward<Receiver>(receiver), std::move(error), stopped);
  }

  /// Sender factory: completes with an open `submit_scope` (cmd begun, ready to record).
  struct enter_submit_scope_sender
  {
    using sender_concept = ex::sender_t;
    using completion_signatures = ex::completion_signatures<ex::set_value_t(submit_scope),
      ex::set_error_t(std::exception_ptr),
      ex::set_stopped_t()>;

    context *ctx{ nullptr };

    [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

    template<class Receiver> struct op_state
    {
      context *ctx{ nullptr };
      Receiver receiver;

      auto start() noexcept -> void
      {
        Receiver rcvr = std::move(receiver);
        auto const token = ex::get_stop_token(ex::get_env(rcvr));
        if constexpr (!ex::unstoppable_token<std::remove_cvref_t<decltype(token)>>) {
          if (token.stop_requested()) {
            ex::set_stopped(std::move(rcvr));
            return;
          }
        }

        std::exception_ptr error;
        std::optional<submit_scope> scope;
        VKEXEC_TRY
        {
          // cppcheck-suppress throwInNoexceptFunction
          scope.emplace(submit_scope::open(*ctx));
        }
        VKEXEC_CATCH_ALL { error = std::current_exception(); }
        if (error) {
          ex::set_error(std::move(rcvr), error);
          return;
        }
        ex::set_value(std::move(rcvr), std::move(*scope));
      }
    };

    // cppcheck-suppress functionStatic
    template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver) -> op_state<Receiver>
    { return op_state<Receiver>{ self.ctx, std::move(receiver) }; }
  };

  [[nodiscard]] inline auto enter_submit_scope(context &ctx) -> enter_submit_scope_sender
  { return enter_submit_scope_sender{ .ctx = &ctx }; }

}// namespace detail

}// namespace vkexec

#endif// VKEXEC_DETAIL_SUBMIT_SCOPE_HPP
