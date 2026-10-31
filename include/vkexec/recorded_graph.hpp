#ifndef VKEXEC_RECORDED_GRAPH_HPP
#define VKEXEC_RECORDED_GRAPH_HPP

#include <vkexec/detail/pass_execution.hpp>

namespace vkexec {

//! Owns recording resources until explicit retirement. Does not own Vulkan commands.
class recorded_graph
{
  struct storage
  {
    detail::context_handle state;
    detail::pass_cleanup cleanup;
    explicit storage(detail::context_handle value) : state(std::move(value)) {}
    virtual ~storage() = default;
    virtual auto finish() -> void = 0;
    auto release() noexcept -> void
    {
#if VKEXEC_HAS_EXCEPTIONS
      try {
#endif
        auto ctx = detail::context_access::facade(state);
        cleanup.release(ctx);
#if VKEXEC_HAS_EXCEPTIONS
      } catch (...) {
        std::terminate();
      }
#endif
    }
  };
  template<class Steps> struct model final : storage
  {
    Steps steps;
    model(detail::context_handle state, Steps value) : storage(std::move(state)), steps(std::move(value)) {}
    auto finish() -> void override
    {
      detail::run_after_gpu(steps);
      detail::commit_after_gpu(steps);
    }
  };
  std::unique_ptr<storage> storage_;
  enum class lifecycle { retired, recorded, submitted };
  lifecycle lifecycle_{ lifecycle::retired };

public:
  recorded_graph() = default;
  recorded_graph(recorded_graph const &) = delete;
  auto operator=(recorded_graph const &) -> recorded_graph & = delete;
  recorded_graph(recorded_graph &&other) noexcept
    : storage_(std::move(other.storage_)), lifecycle_(std::exchange(other.lifecycle_, lifecycle::retired))
  {}
  auto operator=(recorded_graph &&other) noexcept -> recorded_graph &
  {
    if (this != &other) {
      require_retired();
      storage_ = std::move(other.storage_);
      lifecycle_ = std::exchange(other.lifecycle_, lifecycle::retired);
    }
    return *this;
  }
  ~recorded_graph() { require_retired(); }

  //! Required after successful submission, including synchronous submit-and-wait.
  //! Prevents abandon(); the caller must subsequently confirm completion.
  [[nodiscard]] auto mark_submitted() -> status
  {
    if (lifecycle_ != lifecycle::recorded) {
      return fail(errc::invalid_argument, "recording is not awaiting submission");
    }
    lifecycle_ = lifecycle::submitted;
    return {};
  }

  //! Precondition: GPU use has completed. Runs success callbacks exactly once.
  [[nodiscard]] auto complete() -> status
  {
    if (lifecycle_ != lifecycle::submitted) {
      return fail(errc::invalid_argument, "complete requires a submitted recording");
    }
    auto owned = std::move(storage_);
    lifecycle_ = lifecycle::retired;
#if VKEXEC_HAS_EXCEPTIONS
    try {
#endif
      owned->finish();
#if VKEXEC_HAS_EXCEPTIONS
    } catch (...) {
      owned->release();
      return unexpected(unexpected_exception_error());
    }
#endif
    owned->release();
    return {};
  }

  //! Precondition: caller discarded/reset the command buffer without submitting it.
  [[nodiscard]] auto abandon() -> status
  {
    if (lifecycle_ != lifecycle::recorded) {
      return fail(errc::invalid_argument, "abandon requires an unsubmitted recording");
    }
    auto owned = std::move(storage_);
    lifecycle_ = lifecycle::retired;
    owned->release();
    return {};
  }

  template<class Steps>
  [[nodiscard]] static auto
    record(context &ctx, VkCommandBuffer cmd, queue_ref queue, detail::prepared_graph const &plan, Steps steps)
      -> result<recorded_graph>
  {
    auto owned = std::make_unique<model<Steps>>(detail::context_access::state(ctx), std::move(steps));
    status recorded;
#if VKEXEC_HAS_EXCEPTIONS
    try {
#endif
      recorded = detail::record_external(ctx, cmd, queue, plan, owned->steps, owned->cleanup);
#if VKEXEC_HAS_EXCEPTIONS
    } catch (...) {
      owned->release();
      return unexpected(unexpected_exception_error());
    }
#endif
    if (!recorded) {
      owned->release();
      return fail(recorded);
    }
    recorded_graph result;
    result.storage_ = std::move(owned);
    result.lifecycle_ = lifecycle::recorded;
    return result;
  }

private:
  auto require_retired() const noexcept -> void
  {
    assert(!storage_ && "recorded_graph requires complete() or abandon() before destruction/assignment");
    if (storage_) { std::terminate(); }
  }
};

}// namespace vkexec
#endif
