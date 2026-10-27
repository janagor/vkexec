#ifndef VKEXEC_DETAIL_PASS_GRAPH_SUBMIT_HPP
#define VKEXEC_DETAIL_PASS_GRAPH_SUBMIT_HPP

#include <vkexec/detail/pass_execution.hpp>

namespace vkexec {

using pass_graph_completion_signatures =
  stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(error), stdexec::set_stopped_t()>;

namespace detail {

  template<class StepStorage, class Receiver> struct pass_graph_after_gpu_receiver
  {
    using receiver_concept = stdexec::receiver_t;
    Receiver *rcvr{ nullptr };
    StepStorage *steps{ nullptr };

    auto set_value() && noexcept -> void
    {
#if VKEXEC_HAS_EXCEPTIONS
      try {
#endif
        detail::run_after_gpu(*steps);
        detail::commit_after_gpu(*steps);
#if VKEXEC_HAS_EXCEPTIONS
      } catch (...) {
        stdexec::set_error(std::move(*rcvr), unexpected_exception_error());
        return;
      }
#endif
      stdexec::set_value(std::move(*rcvr));
    }

    auto set_error(error err) && noexcept -> void { stdexec::set_error(std::move(*rcvr), std::move(err)); }
    auto set_stopped() && noexcept -> void { stdexec::set_stopped(std::move(*rcvr)); }
    [[nodiscard]] auto get_env() const noexcept -> decltype(stdexec::get_env(std::declval<Receiver const &>()))
    { return stdexec::get_env(std::as_const(*rcvr)); }
  };

  [[nodiscard]] constexpr auto
    is_borrowed_graph_fence(std::size_t index, std::size_t scope_count, bool has_presentation) noexcept -> bool
  { return has_presentation && index + std::size_t{ 1 } == scope_count; }

  struct graph_submit_sender
  {
    using sender_concept = stdexec::sender_t;
    using completion_signatures = pass_graph_completion_signatures;

    std::vector<submit_scope> scopes;
    execution_plan plan;
    std::optional<graph_presentation> presentation;

    // cppcheck-suppress functionStatic
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    [[nodiscard]] auto get_env() const noexcept -> domain_env { return {}; }

    template<class Receiver> struct op_state
    {
      std::vector<submit_scope> scopes;
      execution_plan plan;
      Receiver receiver;
      std::vector<VkSemaphore> semaphores;
      struct binary_edge
      {
        std::size_t source{};
        std::size_t destination{};
        VkSemaphore semaphore{ VK_NULL_HANDLE };
      };
      std::vector<binary_edge> edges;
      std::vector<VkFence> fences;
      bool timeline{};
      static constexpr std::uint64_t k_timeline_signal_value = 1;
      std::optional<graph_presentation> presentation;
      bool presented{};

      auto release(VkDevice device) noexcept -> void
      {
        for (std::size_t index{}; index < fences.size(); ++index) {
          VkFence fence{ fences.at(index) };
          if (fence != VK_NULL_HANDLE && !is_borrowed_graph_fence(index, scopes.size(), presentation.has_value())) {
            vkDestroyFence(device, fence, nullptr);
          }
        }
        fences.clear();
        for (VkSemaphore semaphore : semaphores) {
          if (semaphore != VK_NULL_HANDLE) { vkDestroySemaphore(device, semaphore, nullptr); }
        }
        semaphores.clear();
        scopes.clear();
      }

      [[nodiscard]] auto create_sync(context &facade) -> status
      {
        timeline = facade.capabilities().timeline_semaphore && scopes.size() > std::size_t{ 1 };
        for (std::size_t destination{}; destination < plan.batches.size(); ++destination) {
          std::ranges::transform(plan.batches.at(destination).predecessors,
            std::back_inserter(edges),
            [destination](std::size_t source) -> binary_edge {
              return binary_edge{ .source = source, .destination = destination };
            });
        }
        semaphores.reserve(timeline ? scopes.size() : edges.size());
        fences.reserve(scopes.size());
        for (std::size_t index{}; index < scopes.size(); ++index) {
          if (presentation && index + std::size_t{ 1 } == scopes.size()) {
            fences.push_back(presentation->sync.fence);
            continue;
          }
          VkFence fence{ VK_NULL_HANDLE };
          VkFenceCreateInfo fence_info{};
          fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
          if (VkResult const created = vkCreateFence(facade.device(), &fence_info, nullptr, &fence);
            created != VK_SUCCESS) {
            return fail(created, "vkCreateFence failed");
          }
          fences.push_back(fence);
        }
        auto const semaphore_count = timeline ? scopes.size() : edges.size();
        for (std::size_t index{}; index < semaphore_count; ++index) {
          VkSemaphore semaphore{ VK_NULL_HANDLE };
          VkSemaphoreTypeCreateInfo type_info{};
          type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
          type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
          VkSemaphoreCreateInfo semaphore_info{};
          semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
          semaphore_info.pNext = timeline ? &type_info : nullptr;
          if (VkResult const created = vkCreateSemaphore(facade.device(), &semaphore_info, nullptr, &semaphore);
            created != VK_SUCCESS) {
            return fail(created, "vkCreateSemaphore failed");
          }
          semaphores.push_back(semaphore);
        }
        if (!timeline) {
          for (std::size_t index{}; index < edges.size(); ++index) { edges.at(index).semaphore = semaphores.at(index); }
        }
        return {};
      }

      [[nodiscard]] auto submit_batch(context &facade, std::size_t index) -> status
      {
        auto const &scope = scopes.at(index);
        std::array<VkCommandBuffer, 1> const commands{ scope.cmd };
        std::vector<semaphore_submit> waits;
        std::vector<semaphore_submit> signals;
        for (auto const &edge : edges) {
          if (edge.destination == index) {
            waits.push_back(semaphore_submit{ .semaphore = timeline ? semaphores.at(edge.source) : edge.semaphore,
              .value = timeline ? k_timeline_signal_value : std::uint64_t{} });
          }
          if (!timeline && edge.source == index) { signals.push_back(semaphore_submit{ .semaphore = edge.semaphore }); }
        }
        if (timeline
            && std::ranges::any_of(edges, [index](binary_edge const &edge) -> bool { return edge.source == index; })) {
          signals.push_back(semaphore_submit{ .semaphore = semaphores.at(index), .value = k_timeline_signal_value });
        }
        if (presentation && index + std::size_t{ 1 } == scopes.size()) {
          waits.push_back(presentation->sync.image_available_wait);
          signals.push_back(presentation->sync.render_finished_signal);
          VkFence fence{ fences.at(index) };
          if (VkResult const reset = vkResetFences(facade.device(), 1, &fence); reset != VK_SUCCESS) {
            return fail(reset, "vkResetFences failed for graph presentation");
          }
        }
        return facade.submit(queue_submit{
          .command_buffers = commands,
          .waits = waits,
          .signals = signals,
          .fence = fences.at(index),
          .queue = scope.queue.queue,
        });
      }

      auto start() noexcept -> void
      {
        if (detail::receiver_stop_requested(receiver)) {
          if (presentation) {
            if (auto abandoned = presentation->abort(presentation_abort_state::acquired_only); !abandoned) {
              stdexec::set_error(std::move(receiver), std::move(abandoned.error()));
              return;
            }
          }
          stdexec::set_stopped(std::move(receiver));
          return;
        }
        auto facade = context_access::facade(scopes.front().state);
        auto *device = facade.device();
        std::size_t submitted{};
#if VKEXEC_HAS_EXCEPTIONS
        try {
#endif
          if (auto created = create_sync(facade); !created) {
            finish_error(facade, submitted, std::move(created.error()));
            return;
          }
          for (std::size_t index{}; index < scopes.size(); ++index) {
            auto submitted_status = submit_batch(facade, index);
            if (!submitted_status) {
              finish_error(facade, submitted, std::move(submitted_status.error()));
              return;
            }
            ++submitted;
          }
          if (presentation) {
            if (auto present_result = presentation->present(); !present_result) {
              finish_error(facade, submitted, std::move(present_result.error()));
              return;
            }
            this->presented = true;
          }
          auto const token = detail::receiver_stop_token(receiver);
          VkFence terminal_fence{ fences.back() };
          fences.back() = VK_NULL_HANDLE;
          status enqueued;
#if VKEXEC_HAS_EXCEPTIONS
          try {
#endif
            auto completed = [this, device](std::optional<error> wait_error, bool stopped) mutable noexcept -> void {
              release(device);
              detail::complete_after_reclaim(std::move(receiver), std::move(wait_error), stopped);
            };
            if (presentation) {
              enqueued = facade.enqueue_borrowed_fence_wait(terminal_fence, token, std::move(completed));
            } else {
              enqueued = facade.enqueue_fence_wait(VK_NULL_HANDLE, terminal_fence, token, std::move(completed));
            }
#if VKEXEC_HAS_EXCEPTIONS
          } catch (...) {
            fences.back() = terminal_fence;
            throw;
          }
#endif
          if (!enqueued) { return; }
#if VKEXEC_HAS_EXCEPTIONS
        } catch (...) {
          finish_error(facade, submitted, unexpected_exception_error());
        }
#endif
      }

      auto finish_error(context &facade, std::size_t submitted, error failure) noexcept -> void
      {
        if (submitted != 0) {
          auto const count = static_cast<std::uint32_t>(submitted);
          constexpr std::uint32_t k_retirement_attempts = 3;
          bool retired = false;
          for (std::uint32_t attempt = 0; attempt < k_retirement_attempts; ++attempt) {
            VkResult const waited = vkWaitForFences(facade.device(), count, fences.data(), VK_TRUE, UINT64_MAX);
            if (waited == VK_SUCCESS || waited == VK_ERROR_DEVICE_LOST) {
              retired = true;
              break;
            }
          }
          if (!retired) { std::terminate(); }
        }
        if (presentation && !presented) {
          auto const abort_state = submitted == scopes.size() ? presentation_abort_state::final_submit_completed
                                                              : presentation_abort_state::acquired_only;
          if (auto abandoned = presentation->abort(abort_state); !abandoned) { failure = std::move(abandoned.error()); }
        }
        release(facade.device());
        stdexec::set_error(std::move(receiver), std::move(failure));
      }
    };

    template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) && -> op_state<Receiver>
    {
      return op_state<Receiver>{
        .scopes = std::move(scopes),
        .plan = std::move(plan),
        .receiver = std::move(receiver),
        .semaphores = {},
        .edges = {},
        .fences = {},
        .timeline = false,
        .presentation = std::move(presentation),
        .presented = false,
      };
    }
  };

  template<class StepStorage, class Receiver> struct pass_graph_op_state
  {
    context_handle state;
    StepStorage steps;
    std::vector<queue_affinity> step_queues;
    std::vector<std::vector<std::size_t>> step_predecessors;
    std::optional<graph_presentation> presentation;
    Receiver receiver;
    using child_receiver_t = pass_graph_after_gpu_receiver<StepStorage, Receiver>;
    using fence_sender_t = detail::graph_submit_sender;
    using completion_sender_t =
      decltype(stdexec::continues_on(std::declval<fence_sender_t>(), std::declval<scheduler>()));
    using submit_op_t =
      decltype(stdexec::connect(std::declval<completion_sender_t>(), std::declval<child_receiver_t>()));

    struct submit_op_holder
    {
      submit_op_t op;
      submit_op_holder(detail::graph_submit_sender sender, scheduler sched, child_receiver_t child)
        : op(stdexec::connect(stdexec::continues_on(std::move(sender), std::move(sched)), std::move(child)))
      {}
      ~submit_op_holder() = default;
      submit_op_holder(submit_op_holder const &) = delete;
      auto operator=(submit_op_holder const &) -> submit_op_holder & = delete;
      submit_op_holder(submit_op_holder &&) = delete;
      auto operator=(submit_op_holder &&) -> submit_op_holder & = delete;
    };

    std::optional<submit_op_holder> submit_op;

    auto fail_before_submit(error failure) noexcept -> void
    {
      if (presentation) {
        if (auto abandoned = presentation->abort(presentation_abort_state::acquired_only); !abandoned) {
          failure = std::move(abandoned.error());
        }
      }
      stdexec::set_error(std::move(receiver), std::move(failure));
    }

    [[nodiscard]] auto plan_graph(context &facade) const -> result<execution_plan>
    {
      if (step_predecessors.empty()) {
        return detail::plan_execution(detail::pass_step_count(steps), step_queues, facade.compute_queue_ref());
      }
      return detail::plan_dag(step_queues, step_predecessors, facade.compute_queue_ref());
    }

    [[nodiscard]] auto prepare_presentation(execution_plan const &plan) -> status
    {
      if (!presentation) { return {}; }
      if (plan.batches.empty() || !detail::same_queue(plan.batches.back().queue, presentation->queue)) {
        return fail(errc::invalid_argument, "presentation requires a final batch on the presenter's graphics queue");
      }
      auto sync = presentation->prepare();
      if (!sync) { return fail(sync); }
      presentation->sync = expected_take(sync);
      if (presentation->sync.fence == VK_NULL_HANDLE
          || presentation->sync.image_available_wait.semaphore == VK_NULL_HANDLE
          || presentation->sync.render_finished_signal.semaphore == VK_NULL_HANDLE) {
        return fail(errc::invalid_argument, "presentation returned incomplete synchronization");
      }
      return {};
    }

    [[nodiscard]] auto prepare_terminal(context &facade, execution_plan &plan, std::vector<submit_scope> &scopes)
      -> status
    {
      if (step_predecessors.empty()) { return {}; }
      auto sinks = detail::terminal_batches(plan);
      if (presentation) {
        auto const final_index = plan.batches.size() - std::size_t{ 1 };
        for (std::size_t const index : sinks) {
          if (index != final_index) { plan.batches.back().predecessors.push_back(index); }
        }
        return {};
      }
      if (sinks.size() <= std::size_t{ 1 }) { return {}; }
      detail::execution_batch join{ .queue = facade.compute_queue_ref(),
        .first_step = detail::pass_step_count(steps),
        .end_step = detail::pass_step_count(steps),
        .predecessors = std::move(sinks) };
      auto opened = submit_scope::open(facade, join.queue);
      if (!opened) { return fail(opened); }
      scopes.push_back(expected_take(opened));
      VKEXEC_TRY(scopes.back().end_recording());
      plan.batches.push_back(std::move(join));
      return {};
    }

    auto start() noexcept -> void
    {
      if (detail::receiver_stop_requested(receiver)) {
        if (presentation) {
          if (auto abandoned = presentation->abort(presentation_abort_state::acquired_only); !abandoned) {
            stdexec::set_error(std::move(receiver), std::move(abandoned.error()));
            return;
          }
        }
        stdexec::set_stopped(std::move(receiver));
        return;
      }
#if VKEXEC_HAS_EXCEPTIONS
      try {
#endif
        if (!state) {
          fail_before_submit(make_error(errc::invalid_argument, "pass graph requires a context"));
          return;
        }
        auto facade = context_access::facade(state);
        auto planned = plan_graph(facade);
        if (!planned) {
          fail_before_submit(std::move(planned.error()));
          return;
        }
        if (auto ready = prepare_presentation(*planned); !ready) {
          fail_before_submit(std::move(ready.error()));
          return;
        }
        auto prepared =
          detail::record_execution(state, *planned, steps, presentation ? presentation->image : VK_NULL_HANDLE);
        if (!prepared) {
          fail_before_submit(std::move(prepared.error()));
          return;
        }
        auto scopes = expected_take(prepared);
        if (auto terminal = prepare_terminal(facade, *planned, scopes); !terminal) {
          fail_before_submit(std::move(terminal.error()));
          return;
        }
        auto &child = submit_op.emplace(
          detail::graph_submit_sender{
            .scopes = std::move(scopes), .plan = std::move(*planned), .presentation = std::move(presentation) },
          scheduler_access::make(state),
          child_receiver_t{ .rcvr = &receiver, .steps = &steps });
        stdexec::start(child.op);
#if VKEXEC_HAS_EXCEPTIONS
      } catch (...) {
        fail_before_submit(unexpected_exception_error());
      }
#endif
    }
  };

}// namespace detail

} // namespace vkexec

#endif// VKEXEC_DETAIL_PASS_GRAPH_SUBMIT_HPP
