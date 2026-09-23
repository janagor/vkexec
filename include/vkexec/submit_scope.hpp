#ifndef VKEXEC_SUBMIT_SCOPE_HPP
#define VKEXEC_SUBMIT_SCOPE_HPP

//! \file
//! Command-buffer and descriptor loans for one GPU submit, plus enter/submit senders.

#include <stdexec/execution.hpp>
#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/scheduler.hpp>
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

  /**
   * Descriptor sets allocated for one recording/submit, freed together on scope exit.
   *
   * Tracks both pipeline-keyed sets and raw pool/set pairs from one-off
   * allocations so `release` can free everything after submit.
   */
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

    std::unordered_map<handles::compute_pipeline *, pipeline_set_entry> sets;
    std::vector<allocated_set> allocated;

    //! Frees all tracked descriptor sets on `ctx`'s device.
    auto release(context const &ctx) noexcept -> void;

    //! Records a pool/set pair for later release.
    auto track(VkDescriptorPool pool, VkDescriptorSet set) -> void
    { allocated.push_back(allocated_set{ .pool = pool, .set = set }); }
  };

  //! Compatibility alias used by pass graph recording.
  using pass_cleanup = descriptor_cleanup;

  //! Returns whether two storage-binding lists refer to the same buffers in order.
  auto storage_bindings_equal(std::span<storage_binding const> lhs, std::span<storage_binding const> rhs) -> bool;

  /**
   * Allocates a compute descriptor set from `pipe`'s pool and writes `buffers`.
   *
   * @param ctx Context that owns the device.
   * @param pipe Pipeline whose layout and pool are used.
   * @param buffers Storage bindings matching the pipeline layout.
   */
  auto allocate_compute_set(context const &ctx,
    handles::compute_pipeline &pipe,
    std::span<storage_binding const> buffers) -> result<VkDescriptorSet>;

  /**
   * Reuses a tracked set for `pipe` when bindings match; otherwise allocates a new one.
   *
   * @param cleanup Receives ownership of newly allocated sets.
   */
  auto bind_or_allocate_set(context const &ctx,
    handles::compute_pipeline &pipe,
    std::span<storage_binding const> buffers,
    descriptor_cleanup &cleanup) -> result<VkDescriptorSet>;

  /**
   * Command buffer and descriptor loans for one GPU submit.
   *
   * Exit (destructor or `release`) always frees both. Open with `open`, record
   * into `cmd`, then submit via `submit_and_wait` or `submit_fence` senders.
   *
   * @see enter_submit_scope, submit_and_wait, submit_fence
   */
  struct submit_scope
  {
    context *ctx{ nullptr };
    VkCommandBuffer cmd{ VK_NULL_HANDLE };
    descriptor_cleanup cleanup{};

    submit_scope() = default;
    submit_scope(submit_scope const &) = delete;
    auto operator=(submit_scope const &) -> submit_scope & = delete;

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

    ~submit_scope() { release(); }

    /**
     * Allocates a command buffer from `host`, begins recording, and returns an open scope.
     *
     * @param host Context that owns the command pool.
     */
    [[nodiscard]] static auto open(context &host) -> result<submit_scope>;

    //! Ends Vulkan recording on `cmd` (`vkEndCommandBuffer`).
    // NOLINTNEXTLINE(readability-make-member-function-const) -- ends Vulkan recording; not logically const
    [[nodiscard]] auto end_recording() -> status;

    //! Tracks `set` (from `pipe`'s pool) for release with this scope.
    auto track_set(handles::compute_pipeline const &pipe, VkDescriptorSet set) -> void
    { cleanup.track(pipe.descriptor_pool, set); }

    //! Frees the command buffer and descriptor loans; safe to call more than once.
    auto release() noexcept -> void;
  };

  /**
   * Waits for GPU work then destroys `semaphore`/`fence`.
   *
   * Safe when handles are null. Uses `fallback_queue` only if a queue wait is required.
   */
  auto reclaim_submission_sync(VkDevice device, VkQueue fallback_queue, VkSemaphore semaphore, VkFence fence) noexcept
    -> void;

  /**
   * Delivers exactly one completion signal to `receiver`.
   *
   * Call only after GPU/host sync objects for this submit have already been reclaimed
   * and command/descriptor loans released (or about to be via `release_scope_and_complete`).
   */
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

  /**
   * Sender that completes with an open `submit_scope` (command buffer begun, ready to record).
   *
   * @see enter_submit_scope
   */
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
        auto const token = ex::get_stop_token(ex::get_env(receiver));
        if constexpr (!ex::unstoppable_token<std::remove_cvref_t<decltype(token)>>) {
          if (token.stop_requested()) {
            ex::set_stopped(std::move(receiver));
            return;
          }
        }

        result<submit_scope> opened;
#if VKEXEC_ENABLE_EXCEPTIONS
        try {
          opened = submit_scope::open(*ctx);
        } catch (...) {
          ex::set_error(std::move(receiver), unexpected_exception_error());
          return;
        }
#else
        opened = submit_scope::open(*ctx);
#endif
        if (!opened) {
          ex::set_error(std::move(receiver), std::move(opened.error()));
          return;
        }
        ex::set_value(std::move(receiver), expected_take(opened));
      }
    };

    template<class Receiver>
    // cppcheck-suppress functionStatic
    [[nodiscard]] auto connect(Receiver receiver) & -> op_state<Receiver>
    { return op_state<Receiver>{ ctx, std::move(receiver) }; }

    template<class Receiver>
    // cppcheck-suppress functionStatic
    [[nodiscard]] auto connect(Receiver receiver) && -> op_state<Receiver>
    { return op_state<Receiver>{ ctx, std::move(receiver) }; }
  };

  //! Returns a sender that opens a `submit_scope` on `ctx` (must be non-null).
  [[nodiscard]] auto enter_submit_scope(context *ctx) -> enter_submit_scope_sender;

  //! Returns a sender that opens a `submit_scope` on `ctx`.
  [[nodiscard]] auto enter_submit_scope(context &ctx) -> enter_submit_scope_sender;

  /**
   * Sender that submits a fully recorded scope, blocks until the GPU finishes,
   * then releases command/descriptor loans.
   */
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
        status submitted;
#if VKEXEC_ENABLE_EXCEPTIONS
        try {
          submitted = host->submit_and_wait(scope.cmd);
        } catch (...) {
          scope.release();
          ex::set_error(std::move(receiver), unexpected_exception_error());
          return;
        }
#else
        submitted = host->submit_and_wait(scope.cmd);
#endif
        if (!submitted) {
          scope.release();
          ex::set_error(std::move(receiver), std::move(submitted.error()));
          return;
        }
        scope.release();
        ex::set_value(std::move(receiver));
      }
    };

    template<class Receiver>
    // cppcheck-suppress functionStatic
    // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    [[nodiscard]] auto connect(Receiver receiver) & -> op_state<Receiver>
    {
      return op_state<Receiver>{
        scope,
        std::move(receiver),
      };
    }

    template<class Receiver>
    // cppcheck-suppress functionStatic
    [[nodiscard]] auto connect(Receiver receiver) && -> op_state<Receiver>
    {
      return op_state<Receiver>{
        std::move(scope),
        std::move(receiver),
      };
    }
  };

  //! Builds a blocking submit sender that takes ownership of `scope`.
  [[nodiscard]] auto submit_and_wait(submit_scope scope) -> submit_and_wait_sender;

  /**
   * Sender that submits a recorded scope without blocking `start()`.
   *
   * Completion runs on the context fence agent after reclaiming semaphore/fence
   * and releasing loans. Honours stop tokens with `set_stopped`.
   */
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
        auto const token = ex::get_stop_token(ex::get_env(receiver));
        if constexpr (!ex::unstoppable_token<std::remove_cvref_t<decltype(token)>>) {
          if (token.stop_requested()) {
            scope.release();
            ex::set_stopped(std::move(receiver));
            return;
          }
        }

        context *const host = scope.ctx;
        VkFence fence{ VK_NULL_HANDLE };
        VkSemaphore done{ VK_NULL_HANDLE };
        bool submitted_to_gpu = false;
#if VKEXEC_ENABLE_EXCEPTIONS
        try {
#endif
          if (auto submitted = host->submit_async(scope.cmd, &done, &fence); !submitted) {
            reclaim_submission_sync(host->device(), host->compute_queue(), done, fence);
            scope.release();
            ex::set_error(std::move(receiver), std::move(submitted.error()));
            return;
          }
          submitted_to_gpu = true;

          auto enqueued = host->enqueue_fence_wait(
            done,
            fence,
            token,
            [this](std::optional<error> wait_error, bool stopped) mutable -> void {
              release_scope_and_complete(scope, std::move(receiver), std::move(wait_error), stopped);
            });
          if (!enqueued) {
            // enqueue_fence_wait has already reclaimed the submission and synchronously completed the receiver.
            return;
          }
#if VKEXEC_ENABLE_EXCEPTIONS
        } catch (...) {
          if (submitted_to_gpu) {
            reclaim_submission_sync(host->device(), host->compute_queue(), done, fence);
          }
          scope.release();
          ex::set_error(std::move(receiver), unexpected_exception_error());
        }
#endif
      }
    };

    template<class Receiver>
    // cppcheck-suppress functionStatic
    // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    [[nodiscard]] auto connect(Receiver receiver) & -> op_state<Receiver>
    {
      return op_state<Receiver>{
        scope,
        std::move(receiver),
      };
    }

    template<class Receiver>
    // cppcheck-suppress functionStatic
    [[nodiscard]] auto connect(Receiver receiver) && -> op_state<Receiver>
    {
      return op_state<Receiver>{
        std::move(scope),
        std::move(receiver),
      };
    }
  };

  //! Builds a non-blocking fence-wait submit sender that takes ownership of `scope`.
  [[nodiscard]] auto submit_fence(submit_scope scope) -> submit_fence_sender;

}// namespace detail

}// namespace vkexec

#endif// VKEXEC_SUBMIT_SCOPE_HPP
