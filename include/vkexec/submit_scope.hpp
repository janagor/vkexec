#ifndef VKEXEC_SUBMIT_SCOPE_HPP
#define VKEXEC_SUBMIT_SCOPE_HPP

#include <vkexec/context.hpp>
#include <vkexec/detail/result.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/scheduler.hpp>
#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
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
      std::vector<storage_binding> buffers;
      VkDescriptorSet set{ VK_NULL_HANDLE };
    };

    std::unordered_map<pipeline_resources *, pipeline_set_entry> sets;
    std::vector<allocated_set> allocated;

    auto release(context const &ctx) noexcept -> void;

    auto track(VkDescriptorPool pool, VkDescriptorSet set) -> void
    { allocated.push_back(allocated_set{ .pool = pool, .set = set }); }
  };

  // Compatibility alias used by pass graph recording.
  using pass_cleanup = descriptor_cleanup;

  auto storage_bindings_equal(std::span<storage_binding const> lhs, std::span<storage_binding const> rhs) -> bool;

  auto write_storage_descriptors(VkDevice device, VkDescriptorSet set, std::span<storage_binding const> buffers)
    -> void;

  auto allocate_compute_set(context const &ctx,
    pipeline_resources &pipe,
    std::span<storage_binding const> buffers) -> detail::result<VkDescriptorSet>;

  auto bind_or_allocate_set(context const &ctx,
    pipeline_resources &pipe,
    std::span<storage_binding const> buffers,
    descriptor_cleanup &cleanup) -> detail::result<VkDescriptorSet>;

  /// Command buffer + descriptor loans for one GPU submit. Exit always frees both.
  struct submit_scope
  {
    context *ctx{ nullptr };
    VkCommandBuffer cmd{ VK_NULL_HANDLE };
    descriptor_cleanup cleanup{};

    submit_scope() = default;
    submit_scope(submit_scope const &) = delete;
    auto operator=(submit_scope const &) -> submit_scope & = delete;

    // NOLINTBEGIN(clang-analyzer-core.uninitialized.Assign)
    // CSA cannot model values stored in Boost.LEAF result<>; move is fine at runtime.
    submit_scope(submit_scope &&other) noexcept : ctx(other.ctx), cmd(other.cmd), cleanup(std::move(other.cleanup))
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
    // NOLINTEND(clang-analyzer-core.uninitialized.Assign)

    ~submit_scope() { release(); }

    [[nodiscard]] static auto open(context &host) -> detail::result<submit_scope>;

    // NOLINTNEXTLINE(readability-make-member-function-const) -- ends Vulkan recording; not logically const
    [[nodiscard]] auto end_recording() -> detail::status;

    auto track_set(pipeline_resources const &pipe, VkDescriptorSet set) -> void
    { cleanup.track(pipe.descriptor_pool, set); }

    auto release() noexcept -> void;
  };

  /// Wait for GPU work, destroy semaphore/fence. Safe when handles are null.
  auto
    reclaim_submission_sync(VkDevice device, VkQueue fallback_queue, VkSemaphore semaphore, VkFence fence) noexcept
    -> void;

  /// Release cmd/descriptors, then deliver exactly one completion signal.
  /// Call only after GPU/host sync objects for this submit have already been reclaimed.
  template<class Receiver>
  auto complete_after_reclaim(Receiver &&receiver, std::optional<error> failure, bool stopped) -> void
  {
    if (failure) {
      ex::set_error(std::forward<Receiver>(receiver), std::move(*failure));
    } else if (stopped) {
      ex::set_stopped(std::forward<Receiver>(receiver));
    } else {
      ex::set_value(std::forward<Receiver>(receiver));
    }
  }

  template<class Receiver>
  auto release_scope_and_complete(submit_scope &scope, Receiver &&receiver, std::optional<error> failure, bool stopped)
    -> void
  {
    scope.release();
    complete_after_reclaim(std::forward<Receiver>(receiver), std::move(failure), stopped);
  }

  /// Sender factory: completes with an open `submit_scope` (cmd begun, ready to record).
  struct enter_submit_scope_sender
  {
    using sender_concept = ex::sender_t;
    using completion_signatures =
      ex::completion_signatures<ex::set_value_t(submit_scope), ex::set_error_t(error), ex::set_stopped_t()>;

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

        auto opened = submit_scope::open(*ctx);
        if (!opened) {
          ex::set_error(std::move(rcvr), std::move(opened.error()));
          return;
        }
        ex::set_value(std::move(rcvr), detail::expected_take(opened));
      }
    };

    // cppcheck-suppress functionStatic
    template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver) -> op_state<Receiver>
    { return op_state<Receiver>{ self.ctx, std::move(receiver) }; }
  };

  [[nodiscard]] auto enter_submit_scope(context *ctx) -> enter_submit_scope_sender;

  [[nodiscard]] auto enter_submit_scope(context &ctx) -> enter_submit_scope_sender;

  /// Submit a fully recorded scope and block until the GPU finishes, then release loans.
  struct submit_and_wait_sender
  {
    using sender_concept = ex::sender_t;
    using completion_signatures = ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error)>;

    submit_scope scope;

    [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = scope.ctx }; }

    template<class Receiver> struct op_state
    {
      submit_scope scope;
      Receiver receiver;

      auto start() noexcept -> void
      {
        context *const host = scope.ctx;
        if (auto submitted = host->submit_and_wait(scope.cmd); !submitted) {
          scope.release();
          ex::set_error(std::move(receiver), std::move(submitted.error()));
          return;
        }
        scope.release();
        ex::set_value(std::move(receiver));
      }
    };

    // cppcheck-suppress functionStatic
    template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver) -> op_state<Receiver>
    {
      return op_state<Receiver>{
        std::forward_like<decltype(self)>(self.scope),
        std::move(receiver),
      };
    }
  };

  [[nodiscard]] auto submit_and_wait(submit_scope scope) -> submit_and_wait_sender;

  /// Submit a recorded scope without blocking `start()`; reclaim then complete on the fence agent.
  struct submit_fence_sender
  {
    using sender_concept = ex::sender_t;
    using completion_signatures =
      ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

    submit_scope scope;

    [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = scope.ctx }; }

    template<class Receiver> struct op_state
    {
      submit_scope scope;
      Receiver receiver;

      auto start() noexcept -> void
      {
        Receiver rcvr = std::move(receiver);
        auto const token = ex::get_stop_token(ex::get_env(rcvr));
        if constexpr (!ex::unstoppable_token<std::remove_cvref_t<decltype(token)>>) {
          if (token.stop_requested()) {
            scope.release();
            ex::set_stopped(std::move(rcvr));
            return;
          }
        }

        context *const host = scope.ctx;
        VkFence fence{ VK_NULL_HANDLE };
        VkSemaphore done{ VK_NULL_HANDLE };
        if (auto submitted = host->submit_async(scope.cmd, &done, &fence); !submitted) {
          reclaim_submission_sync(host->device(), host->compute_queue(), done, fence);
          scope.release();
          ex::set_error(std::move(rcvr), std::move(submitted.error()));
          return;
        }

        if (auto enqueued = host->enqueue_fence_wait(done,
              fence,
              token,
              [scope = std::move(scope), rcvr = std::move(rcvr)](std::optional<error> wait_error, bool stopped) mutable
                -> void { release_scope_and_complete(scope, std::move(rcvr), std::move(wait_error), stopped); });
          !enqueued) {
          // on_done already completed `rcvr` (and released `scope`) on the failure path.
          (void)enqueued;
        }
      }
    };

    // cppcheck-suppress functionStatic
    template<class Receiver> [[nodiscard]] auto connect(this auto &&self, Receiver receiver) -> op_state<Receiver>
    {
      return op_state<Receiver>{
        std::forward_like<decltype(self)>(self.scope),
        std::move(receiver),
      };
    }
  };

  [[nodiscard]] auto submit_fence(submit_scope scope) -> submit_fence_sender;

}// namespace detail

}// namespace vkexec

#endif// VKEXEC_SUBMIT_SCOPE_HPP
