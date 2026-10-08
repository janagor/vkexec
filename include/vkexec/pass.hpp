#ifndef VKEXEC_PASS_HPP
#define VKEXEC_PASS_HPP

//! \file
//! Compute pass recording, pass graphs, and stdexec pipe adaptors.

#include <vkexec/barrier_params.hpp>
#include <vkexec/config.hpp>
#include <vkexec/detail/attributes.hpp>
#include <vkexec/detail/resource_state_tracker.hpp>
#include <vkexec/detail/sender_expr.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/presentation_abort_state.hpp>
#include <vkexec/push.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/result.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit_scope.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#if VKEXEC_HAS_EXCEPTIONS
#include <exception>
#endif
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

namespace ex = stdexec;

//! An unset affinity uses the graph's default queue during execution planning.
using queue_affinity = std::optional<queue_ref>;

//! Synchronization borrowed from a presenter for one graph execution.
struct graph_presentation_sync
{
  semaphore_submit image_available_wait;
  semaphore_submit render_finished_signal;
  VkFence fence{ VK_NULL_HANDLE };
};

//! External swapchain boundary attached to a complete pass graph.
struct graph_presentation
{
  queue_ref queue;
  VkImage image{ VK_NULL_HANDLE };
  std::function<result<graph_presentation_sync>()> prepare;
  std::function<status()> present;
  std::function<status(presentation_abort_state)> abort;
  graph_presentation_sync sync;
};

//! Workgroup counts for `vkCmdDispatch` (X/Y/Z).
struct dispatch
{
  std::uint32_t x{ 1 };
  std::uint32_t y{ 1 };
  std::uint32_t z{ 1 };
};

/**
 * Computes workgroup count X covering `work_count` invocations for local size `local_x`.
 *
 * Y and Z remain 1.
 *
 * @pre local_x > 0
 */
[[nodiscard]] constexpr auto dispatch_groups_for(std::uint32_t work_count, std::uint32_t local_x) noexcept -> dispatch
{ return dispatch{ .x = (work_count / local_x) + static_cast<std::uint32_t>(work_count % local_x != 0U) }; }

//! Arguments for `vkCmdDispatchIndirect`.
struct indirect_dispatch
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  VkDeviceSize offset{ 0 };
};

//! Pipeline, layout, and optional descriptor set for one compute dispatch.
struct compute_bind
{
  VkPipeline pipeline{ VK_NULL_HANDLE };
  VkPipelineLayout layout{ VK_NULL_HANDLE };
  VkDescriptorSet set{ VK_NULL_HANDLE };
  std::vector<image_use> images;
  std::vector<buffer_use> buffers;
  bool resource_metadata{ false };
  bool complete_resource_metadata{ true };
};

//! Builds a `compute_bind` from pipeline resources and an optional set.
//! Raw descriptor sets and descriptor-heap binds have no resource metadata;
//! graph synchronization cannot infer their descriptor accesses.
[[nodiscard]] auto bind_compute(handles::compute_pipeline const &pipe, VkDescriptorSet set = VK_NULL_HANDLE)
  -> compute_bind;

//! Builds a binding that retains the descriptor resources for graph planning.
[[nodiscard]] auto bind_compute(handles::compute_pipeline const &pipe, VkDescriptorSet set, resource_table const &table)
  -> compute_bind;

/**
 * Records bind, optional push constants, and a direct dispatch on `cmd`.
 *
 * @param cmd Command buffer in the recording state.
 * @param bind Pipeline / layout / descriptor set.
 * @param push Push-constant bytes (may be null when `push_bytes` is 0).
 * @param push_bytes Size of the push-constant blob.
 * @param groups Workgroup counts for `vkCmdDispatch`.
 */
auto record_pass(VkCommandBuffer cmd,
  compute_bind const &bind,
  void const *push,
  std::uint32_t push_bytes,
  dispatch groups) -> void;

/**
 * Records bind, optional push constants, and an indirect dispatch on `cmd`.
 *
 * @param groups Buffer and offset containing `VkDispatchIndirectCommand`.
 */
auto record_pass(VkCommandBuffer cmd,
  compute_bind const &bind,
  void const *push,
  std::uint32_t push_bytes,
  indirect_dispatch groups) -> void;

//! Convenience overload that builds a `compute_bind` from `pipe` and `set`.
auto record_pass(VkCommandBuffer cmd,
  handles::compute_pipeline const &pipe,
  VkDescriptorSet set,
  void const *push,
  std::uint32_t push_bytes,
  dispatch groups) -> void;

namespace detail {

  struct no_push_constants
  {
  };

  struct on_queue_t
  {
  };

  struct custom_pass_t
  {
  };

  template<class Uses, class Record> struct custom_pass_data
  {
    VKEXEC_NO_UNIQUE_ADDRESS Uses resources;
    VKEXEC_NO_UNIQUE_ADDRESS Record record_fn;
  };

  template<class Uses, class Record> struct custom_pass_step
  {
    VKEXEC_NO_UNIQUE_ADDRESS Uses resources;
    VKEXEC_NO_UNIQUE_ADDRESS Record record_fn;

    auto record(context & /*ctx*/, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> status
    {
      if constexpr (std::same_as<std::invoke_result_t<Record &, VkCommandBuffer>, status>) {
        return std::invoke(record_fn, cmd);
      } else {
        std::invoke(record_fn, cmd);
        return {};
      }
    }
  };

  template<class T> struct is_byte_span : std::false_type
  {
  };

  template<std::size_t Extent> struct is_byte_span<std::span<std::byte, Extent>> : std::true_type
  {
  };

  template<std::size_t Extent> struct is_byte_span<std::span<std::byte const, Extent>> : std::true_type
  {
  };

  template<class T>
  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr bool is_byte_span_v = is_byte_span<std::remove_cvref_t<T>>::value;

  template<class T>
  concept dispatch_kind =
    std::same_as<std::remove_cvref_t<T>, dispatch> || std::same_as<std::remove_cvref_t<T>, indirect_dispatch>;

  template<class T>
  concept push_constant_type = std::same_as<std::remove_cvref_t<T>, no_push_constants>
                               || (std::is_trivially_copyable_v<std::remove_cvref_t<T>> && !is_byte_span_v<T>);

  template<class Step>
  concept static_pass_step =
    std::move_constructible<Step> && requires(Step &step, context &ctx, VkCommandBuffer cmd, pass_cleanup &cleanup) {
      { step.record(ctx, cmd, cleanup) } -> std::same_as<status>;
    };

  template<class T>
  concept sender_adaptor_closure =
    std::derived_from<std::decay_t<T>, ex::sender_adaptor_closure<std::decay_t<T>>>
    && std::move_constructible<std::decay_t<T>> && std::constructible_from<std::decay_t<T>, T>;

  template<class Step> auto run_after_gpu(Step &step) -> void
  {
    if constexpr (requires {
                    { step.after_gpu() } -> std::same_as<void>;
                  }) {
      step.after_gpu();
    }
  }

  class dynamic_pass_step
  {
    struct interface
    {
      interface() = default;
      virtual ~interface() = default;

      interface(interface const &) = delete;
      auto operator=(interface const &) -> interface & = delete;
      interface(interface &&) = delete;
      auto operator=(interface &&) -> interface & = delete;

      virtual auto record(context &ctx, VkCommandBuffer cmd, pass_cleanup &cleanup) -> status = 0;
      virtual auto after_gpu() -> void = 0;
      virtual auto declare_resources(resource_state_tracker &tracker, std::size_t step, queue_ref queue) -> status = 0;
      [[nodiscard]] virtual auto has_resource_declarations() const noexcept -> bool = 0;
    };

    template<static_pass_step Step> struct model final : interface
    {
      Step step;

      explicit model(Step value) : step(std::move(value)) {}

      auto record(context &ctx, VkCommandBuffer cmd, pass_cleanup &cleanup) -> status override
      { return step.record(ctx, cmd, cleanup); }

      auto after_gpu() -> void override { detail::run_after_gpu(step); }

      auto declare_resources(resource_state_tracker &tracker, std::size_t index, queue_ref queue) -> status override
      {
        if constexpr (requires { step.resources; }) {
          return tracker.use(step.resources, index, queue.family, queue.queue);
        } else if constexpr (requires { step.declare_resources(tracker, index, queue); }) {
          return step.declare_resources(tracker, index, queue);
        }
        return {};
      }

      [[nodiscard]] auto has_resource_declarations() const noexcept -> bool override
      {
        if constexpr (requires { step.has_resource_declarations(); }) { return step.has_resource_declarations(); }
        if constexpr (requires { step.bind.resource_metadata; }) { return step.bind.resource_metadata; }
        return requires { step.resources; };
      }
    };

    std::unique_ptr<interface> impl_;

  public:
    dynamic_pass_step() = delete;

    template<class Step>
      requires(!std::same_as<std::remove_cvref_t<Step>, dynamic_pass_step>) && static_pass_step<Step>
    explicit dynamic_pass_step(Step step) : impl_(std::make_unique<model<Step>>(std::move(step)))
    {}

    ~dynamic_pass_step() = default;
    dynamic_pass_step(dynamic_pass_step &&) noexcept = default;
    auto operator=(dynamic_pass_step &&) noexcept -> dynamic_pass_step & = default;
    dynamic_pass_step(dynamic_pass_step const &) = delete;
    auto operator=(dynamic_pass_step const &) -> dynamic_pass_step & = delete;

    auto record(context &ctx, VkCommandBuffer cmd, pass_cleanup &cleanup) -> status
    { return impl_->record(ctx, cmd, cleanup); }

    auto after_gpu() -> void { impl_->after_gpu(); }

    auto declare_resources(resource_state_tracker &tracker, std::size_t step, queue_ref queue) -> status
    { return impl_->declare_resources(tracker, step, queue); }

    [[nodiscard]] auto has_resource_declarations() const noexcept -> bool { return impl_->has_resource_declarations(); }
  };

  static_assert(std::move_constructible<dynamic_pass_step>);
  static_assert(!std::copy_constructible<dynamic_pass_step>);

  [[nodiscard]] inline auto push_bytes(no_push_constants const & /*unused*/) noexcept -> std::span<std::byte const>
  { return {}; }

  template<class Push>
    requires std::is_trivially_copyable_v<Push>
  [[nodiscard]] auto push_bytes(Push const &push) noexcept -> std::span<std::byte const>
  { return std::as_bytes(std::span{ &push, std::size_t{ 1 } }); }

  template<push_constant_type Push, dispatch_kind Dispatch> struct compute_pass_step
  {
    compute_bind bind{};
    VKEXEC_NO_UNIQUE_ADDRESS Push push{};
    Dispatch dispatch_info{};

    auto declare_resources(resource_state_tracker &tracker, std::size_t step, queue_ref queue) const -> status
    {
      if (!bind.resource_metadata) { return {}; }
      if (!bind.complete_resource_metadata) {
        return fail(errc::invalid_argument, "compute binding lacks resource or layout metadata");
      }
      for (auto const &image : bind.images) { VKEXEC_TRY(tracker.use(image, step, queue.family, queue.queue)); }
      for (auto const &buffer : bind.buffers) { VKEXEC_TRY(tracker.use(buffer, step, queue.family, queue.queue)); }
      if constexpr (std::same_as<Dispatch, indirect_dispatch>) {
        VKEXEC_TRY(tracker.use(
          read(dispatch_info.buffer, dispatch_info.offset, sizeof(VkDispatchIndirectCommand), buffer_usage::indirect),
          step,
          queue.family,
          queue.queue));
      }
      return {};
    }

    auto record(context & /*ctx*/, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> status
    {
      auto const bytes = push_bytes(push);
      void const *const data = bytes.empty() ? nullptr : static_cast<void const *>(bytes.data());
      record_pass(cmd, bind, data, static_cast<std::uint32_t>(bytes.size()), dispatch_info);
      return {};
    }
  };

  template<push_constant_type Push, dispatch_kind Dispatch> struct compute_pass_data
  {
    compute_bind bind{};
    VKEXEC_NO_UNIQUE_ADDRESS Push push{};
    Dispatch dispatch_info{};
  };

  template<class Tag> struct barrier_step
  {
    VKEXEC_NO_UNIQUE_ADDRESS Tag tag;

    auto record(context &ctx, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> status { return tag(ctx, cmd); }
  };

  template<class Tag> [[nodiscard]] auto make_barrier_step(Tag tag) -> barrier_step<Tag>
  { return barrier_step<Tag>{ .tag = std::move(tag) }; }

  template<class Record> struct callback_pass_step
  {
    VKEXEC_NO_UNIQUE_ADDRESS Record record_fn;

    auto record(context &ctx, VkCommandBuffer cmd, pass_cleanup &cleanup) -> status
    { return record_fn(ctx, cmd, cleanup); }
  };

  template<class Record, class AfterGpu> struct callback_after_gpu_pass_step
  {
    VKEXEC_NO_UNIQUE_ADDRESS Record record_fn;
    VKEXEC_NO_UNIQUE_ADDRESS AfterGpu after_gpu_fn;

    auto record(context &ctx, VkCommandBuffer cmd, pass_cleanup &cleanup) -> status
    { return record_fn(ctx, cmd, cleanup); }

    auto after_gpu() -> void { after_gpu_fn(); }
  };

  template<class Record> [[nodiscard]] auto make_callback_pass_step(Record record)
  { return callback_pass_step<Record>{ .record_fn = std::move(record) }; }

  template<class Record, class AfterGpu> [[nodiscard]] auto make_callback_pass_step(Record record, AfterGpu after_gpu)
  {
    return callback_after_gpu_pass_step<Record, AfterGpu>{ .record_fn = std::move(record),
      .after_gpu_fn = std::move(after_gpu) };
  }

  template<class... Steps>
  [[nodiscard]] auto
    record_static_steps(context &ctx, VkCommandBuffer cmd, pass_cleanup &cleanup, std::tuple<Steps...> &steps) -> status
  {
    status result{};
    auto record_one = [&ctx, cmd, &cleanup, &result](auto &step) -> bool {
      auto current = step.record(ctx, cmd, cleanup);
      if (!current) {
        result = fail(current);
        return false;
      }
      return true;
    };
    bool const success = std::apply([&record_one](auto &...step) -> bool { return (record_one(step) && ...); }, steps);
    if (!success) { return result; }
    return {};
  }

  [[nodiscard]] inline auto record_dynamic_steps(context &ctx,
    VkCommandBuffer cmd,
    pass_cleanup &cleanup,
    std::vector<dynamic_pass_step> &steps) -> status
  {
    for (auto &step : steps) {
      auto recorded = step.record(ctx, cmd, cleanup);
      if (!recorded) { return fail(recorded); }
    }
    return {};
  }

  template<class... Steps>
  [[nodiscard]] auto record_pass_steps(submit_scope &scope, std::tuple<Steps...> &steps) -> status
  {
    auto facade = context_access::facade(scope.state);
    if (auto recorded = record_static_steps(facade, scope.cmd, scope.cleanup, steps); !recorded) {
      return fail(recorded);
    }
    return scope.end_recording();
  }

  [[nodiscard]] inline auto record_pass_steps(submit_scope &scope, std::vector<dynamic_pass_step> &steps) -> status
  {
    auto facade = context_access::facade(scope.state);
    if (auto recorded = record_dynamic_steps(facade, scope.cmd, scope.cleanup, steps); !recorded) {
      return fail(recorded);
    }
    return scope.end_recording();
  }

  struct execution_batch
  {
    queue_ref queue{};
    std::size_t first_step{};
    std::size_t end_step{};
  };

  struct execution_plan
  {
    std::vector<execution_batch> batches;
  };

  [[nodiscard]] inline auto same_queue(queue_ref lhs, queue_ref rhs) noexcept -> bool
  { return lhs.queue == rhs.queue && lhs.family == rhs.family; }

  [[nodiscard]] inline auto plan_batches(std::span<queue_affinity const> queues, queue_ref default_queue)
    -> execution_plan
  {
    execution_plan plan;
    std::size_t index{};
    for (queue_affinity const &affinity : queues) {
      queue_ref const queue = affinity.value_or(default_queue);
      if (plan.batches.empty() || !same_queue(plan.batches.back().queue, queue)) {
        plan.batches.push_back(execution_batch{ .queue = queue, .first_step = index, .end_step = index + 1 });
      } else {
        plan.batches.back().end_step = index + 1;
      }
      ++index;
    }
    return plan;
  }

  [[nodiscard]] inline auto plan_execution(std::size_t step_count,
    std::span<queue_affinity const> queues,
    queue_ref default_queue) -> result<execution_plan>
  {
    if (!queues.empty() && queues.size() != step_count) {
      return fail(errc::invalid_argument, "pass queue count mismatch");
    }
    std::vector<queue_affinity> default_queues;
    if (queues.empty()) { default_queues.resize(step_count); }
    return plan_batches(queues.empty() ? std::span<queue_affinity const>{ default_queues } : queues, default_queue);
  }

  template<class StepStorage>
  [[nodiscard]] constexpr auto pass_step_count(StepStorage const &steps) noexcept -> std::size_t
  {
    if constexpr (requires { steps.size(); }) {
      return steps.size();
    } else {
      return std::tuple_size_v<StepStorage>;
    }
  }

  template<class StepStorage>
  [[nodiscard]] auto
    record_batch_steps(context &facade, submit_scope &scope, execution_batch const &batch, StepStorage &steps) -> status
  {
    if constexpr (requires { steps.size(); }) {
      for (std::size_t index = batch.first_step; index < batch.end_step; ++index) {
        if (auto recorded = steps.at(index).record(facade, scope.cmd, scope.cleanup); !recorded) {
          return fail(recorded);
        }
      }
      return {};
    } else {
      status recorded{};
      std::size_t index{};
      auto record_one = [&](auto &step) -> void {
        if (recorded && index >= batch.first_step && index < batch.end_step) {
          recorded = step.record(facade, scope.cmd, scope.cleanup);
        }
        ++index;
      };
      std::apply([&](auto &...step) -> void { (record_one(step), ...); }, steps);
      return recorded;
    }
  }

  [[nodiscard]] inline auto validate_execution_batches(execution_plan const &plan, std::size_t step_count) -> status
  {
    if (plan.batches.empty() && step_count != 0) {
      return fail(errc::invalid_argument, "invalid empty execution plan");
    }
    std::size_t validated_step{};
    for (auto const &batch : plan.batches) {
      if (batch.first_step != validated_step || batch.end_step <= batch.first_step || batch.end_step > step_count) {
        return fail(errc::invalid_argument, "invalid execution batch");
      }
      validated_step = batch.end_step;
    }
    if (validated_step != step_count) { return fail(errc::invalid_argument, "incomplete execution plan"); }
    return {};
  }

  [[nodiscard]] inline auto crosses_queue_families(execution_plan const &execution) -> bool
  {
    for (std::size_t index = 1; index < execution.batches.size(); ++index) {
      if (execution.batches.at(index - 1).queue.family != execution.batches.at(index).queue.family) { return true; }
    }
    return false;
  }

  template<class Step>
  [[nodiscard]] auto declare_step_resources(Step const &step,
    resource_state_tracker &tracker,
    std::size_t index,
    queue_ref queue,
    bool crosses_families) -> status
  {
    if constexpr (requires { step.resources; }) {
      return tracker.use(step.resources, index, queue.family, queue.queue);
    } else if constexpr (requires { step.declare_resources(tracker, index, queue); }) {
      if (crosses_families) {
        if constexpr (requires { step.has_resource_declarations(); }) {
          if (!step.has_resource_declarations()) {
            return fail(errc::unsupported, "cross-family pass steps require resource declarations");
          }
        } else if constexpr (requires { step.bind.resource_metadata; }) {
          if (!step.bind.resource_metadata) {
            return fail(errc::unsupported, "cross-family pass steps require resource declarations");
          }
        } else {
          return fail(errc::unsupported, "cross-family pass steps require resource declarations");
        }
      }
      return step.declare_resources(tracker, index, queue);
    } else {
      if (crosses_families) { return fail(errc::unsupported, "cross-family pass steps require resource declarations"); }
      return {};
    }
  }

  [[nodiscard]] inline auto declare_step_resources(dynamic_pass_step &step,
    resource_state_tracker &tracker,
    std::size_t index,
    queue_ref queue,
    bool crosses_families) -> status
  {
    if (crosses_families && !step.has_resource_declarations()) {
      return fail(errc::unsupported, "cross-family pass steps require resource declarations");
    }
    return step.declare_resources(tracker, index, queue);
  }

  template<class... Steps>
  [[nodiscard]] auto declare_indexed_step(std::tuple<Steps...> const &steps,
    resource_state_tracker &tracker,
    std::size_t index,
    queue_ref queue,
    bool crosses_families) -> status
  {
    status declared{};
    std::size_t position{};
    std::apply(
      [&](auto const &...step) -> void {
        auto declare_one = [&](auto const &current) -> void {
          if (position == index) {
            declared = declare_step_resources(current, tracker, index, queue, crosses_families);
          }
          ++position;
        };
        (declare_one(step), ...);
      },
      steps);
    return declared;
  }

  template<class StepStorage>
  [[nodiscard]] auto plan_resource_sync(execution_plan const &execution, StepStorage &steps)
    -> result<resource_sync_plan>
  {
    resource_state_tracker tracker{ pass_step_count(steps) };
    bool const crosses_families = crosses_queue_families(execution);
    for (auto const &batch : execution.batches) {
      for (std::size_t index = batch.first_step; index < batch.end_step; ++index) {
        status declared{};
        if constexpr (requires { steps.size(); }) {
          declared = declare_step_resources(steps.at(index), tracker, index, batch.queue, crosses_families);
        } else {
          declared = declare_indexed_step(steps, tracker, index, batch.queue, crosses_families);
        }
        if (!declared) { return fail(declared); }
      }
    }
    return std::move(tracker).finish();
  }

  [[nodiscard]] auto record_sync(context &facade,
    VkCommandBuffer cmd,
    std::vector<image_barrier_params> const &images,
    std::vector<buffer_barrier_params> const &buffers) -> status;

  template<class StepStorage>
  [[nodiscard]] auto record_sync_batch(context &facade,
    submit_scope &scope,
    execution_batch const &batch,
    resource_sync_plan const &sync,
    StepStorage &steps) -> status
  {
    for (std::size_t index = batch.first_step; index < batch.end_step; ++index) {
      auto const &point = sync.steps.at(index);
      VKEXEC_TRY(record_sync(facade, scope.cmd, point.images_before, point.buffers_before));
      if constexpr (requires { steps.size(); }) {
        VKEXEC_TRY(steps.at(index).record(facade, scope.cmd, scope.cleanup));
      } else {
        status recorded{};
        std::size_t position{};
        std::apply(
          [&](auto &...step) -> void {
            auto record_one = [&](auto &current) -> void {
              if (position == index) { recorded = current.record(facade, scope.cmd, scope.cleanup); }
              ++position;
            };
            (record_one(step), ...);
          },
          steps);
        VKEXEC_TRY(recorded);
      }
      VKEXEC_TRY(record_sync(facade, scope.cmd, point.images_after, point.buffers_after));
    }
    return {};
  }

  [[nodiscard]] inline auto
    validate_presentation_image(resource_sync_plan const &sync, execution_plan const &plan, VkImage image) -> status
  {
    if (image == VK_NULL_HANDLE) { return {}; }
    bool saw_image{};
    for (auto const &use : sync.image_uses) {
      if (use.image != image) { continue; }
      saw_image = true;
      if (use.step < plan.batches.back().first_step || use.usage != image_usage::color_attachment
          || use.final_layout != VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
        return fail(errc::invalid_argument,
          "presented image must be used as a color attachment only in the final batch and finish in present layout");
      }
    }
    if (!saw_image) { return fail(errc::invalid_argument, "presented image has no declared graph use"); }
    return {};
  }

  template<class StepStorage>
  [[nodiscard]] auto record_execution(context_handle const &state,
    execution_plan const &plan,
    StepStorage &steps,
    VkImage presentation_image = VK_NULL_HANDLE) -> result<std::vector<submit_scope>>
  {
    if (!state) { return fail(errc::invalid_argument, "pass graph requires a context"); }
    auto const step_count = pass_step_count(steps);
    if (auto validated = validate_execution_batches(plan, step_count); !validated) { return fail(validated); }
    auto synchronization = plan_resource_sync(plan, steps);
    if (!synchronization) { return fail(synchronization); }
    VKEXEC_TRY(validate_presentation_image(*synchronization, plan, presentation_image));
    auto facade = context_access::facade(state);
    std::vector<submit_scope> scopes;
    scopes.reserve(plan.batches.empty() ? std::size_t{ 1 } : plan.batches.size());
    for (auto const &batch : plan.batches) {
      auto opened = submit_scope::open(facade, batch.queue);
      if (!opened) { return fail(opened); }
      scopes.push_back(expected_take(opened));
      auto &scope = scopes.back();
      auto recorded = record_sync_batch(facade, scope, batch, *synchronization, steps);
      if (!recorded) { return fail(recorded); }
      if (auto ended = scope.end_recording(); !ended) { return fail(ended); }
    }
    if (scopes.empty()) {
      auto opened = submit_scope::open(facade);
      if (!opened) { return fail(opened); }
      scopes.push_back(expected_take(opened));
      if (auto ended = scopes.back().end_recording(); !ended) { return fail(ended); }
    }
    return scopes;
  }

  template<class... Steps> auto run_after_gpu(std::tuple<Steps...> &steps) -> void
  {
    std::apply([](auto &...step) -> void { (detail::run_after_gpu(step), ...); }, steps);
  }

  inline auto run_after_gpu(std::vector<dynamic_pass_step> &steps) -> void
  {
    for (auto &step : steps) { step.after_gpu(); }
  }

}// namespace detail

using pass_graph_completion_signatures =
  ex::completion_signatures<ex::set_value_t(), ex::set_error_t(error), ex::set_stopped_t()>;

namespace detail {

  template<class StepStorage, class Receiver> struct pass_graph_after_gpu_receiver
  {
    using receiver_concept = ex::receiver_t;
    Receiver *rcvr{ nullptr };
    StepStorage *steps{ nullptr };

    auto set_value() && noexcept -> void
    {
#if VKEXEC_HAS_EXCEPTIONS
      try {
#endif
        detail::run_after_gpu(*steps);
#if VKEXEC_HAS_EXCEPTIONS
      } catch (...) {
        ex::set_error(std::move(*rcvr), unexpected_exception_error());
        return;
      }
#endif
      ex::set_value(std::move(*rcvr));
    }

    auto set_error(error err) && noexcept -> void { ex::set_error(std::move(*rcvr), std::move(err)); }
    auto set_stopped() && noexcept -> void { ex::set_stopped(std::move(*rcvr)); }
    [[nodiscard]] auto get_env() const noexcept -> decltype(ex::get_env(std::declval<Receiver const &>()))
    { return ex::get_env(std::as_const(*rcvr)); }
  };

  struct graph_submit_sender
  {
    using sender_concept = ex::sender_t;
    using completion_signatures = pass_graph_completion_signatures;

    std::vector<submit_scope> scopes;
    std::optional<graph_presentation> presentation;

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    [[nodiscard]] auto get_env() const noexcept -> domain_env { return {}; }

    template<class Receiver> struct op_state
    {
      std::vector<submit_scope> scopes;
      Receiver receiver;
      std::vector<VkSemaphore> semaphores;
      std::vector<VkFence> fences;
      bool timeline{};
      std::optional<graph_presentation> presentation;
      bool presented{};

      auto release(VkDevice device) noexcept -> void
      {
        for (std::size_t index{}; index < fences.size(); ++index) {
          VkFence fence = fences.at(index);
          if (fence != VK_NULL_HANDLE && (!presentation || index + std::size_t{ 1 } != fences.size())) {
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
        auto const boundary_count = scopes.size() - std::size_t{ 1 };
        timeline = facade.capabilities().timeline_semaphore && boundary_count != 0;
        semaphores.reserve(timeline ? std::size_t{ 1 } : boundary_count);
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
        auto const semaphore_count = timeline ? std::size_t{ 1 } : boundary_count;
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
        return {};
      }

      [[nodiscard]] auto submit_batch(context &facade, std::size_t index) -> status
      {
        auto const boundary_count = scopes.size() - std::size_t{ 1 };
        auto const &scope = scopes.at(index);
        std::array<VkCommandBuffer, 1> const commands{ scope.cmd };
        std::array<semaphore_submit, 2> waits{};
        std::array<semaphore_submit, 2> signals{};
        std::size_t wait_count{};
        std::size_t signal_count{};
        if (index != 0) {
          waits.at(wait_count++) = semaphore_submit{
            .semaphore = semaphores.at(timeline ? std::size_t{} : index - std::size_t{ 1 }),
            .value = timeline ? index : std::size_t{},
          };
        }
        if (index < boundary_count) {
          signals.at(signal_count++) = semaphore_submit{
            .semaphore = semaphores.at(timeline ? std::size_t{} : index),
            .value = timeline ? index + std::size_t{ 1 } : std::size_t{},
          };
        }
        if (presentation && index == boundary_count) {
          waits.at(wait_count++) = presentation->sync.image_available_wait;
        }
        if (presentation && index == boundary_count) {
          signals.at(signal_count++) = presentation->sync.render_finished_signal;
          VkFence fence = fences.at(index);
          if (VkResult const reset = vkResetFences(facade.device(), 1, &fence); reset != VK_SUCCESS) {
            return fail(reset, "vkResetFences failed for graph presentation");
          }
        }
        return facade.submit(queue_submit{
          .command_buffers = commands,
          .waits = std::span<semaphore_submit const>{ waits.data(), wait_count },
          .signals = std::span<semaphore_submit const>{ signals.data(), signal_count },
          .fence = fences.at(index),
          .queue = scope.queue.queue,
        });
      }

      auto start() noexcept -> void
      {
        if (detail::receiver_stop_requested(receiver)) {
          if (presentation) {
            if (auto abandoned = presentation->abort(presentation_abort_state::acquired_only); !abandoned) {
              ex::set_error(std::move(receiver), std::move(abandoned.error()));
              return;
            }
          }
          ex::set_stopped(std::move(receiver));
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
          VkFence terminal_fence = fences.back();
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
          (void)vkWaitForFences(facade.device(), count, fences.data(), VK_TRUE, UINT64_MAX);
        }
        if (presentation && !presented) {
          auto const abort_state = submitted == scopes.size() ? presentation_abort_state::final_submit_completed
                                                              : presentation_abort_state::acquired_only;
          if (auto abandoned = presentation->abort(abort_state); !abandoned) { failure = std::move(abandoned.error()); }
        }
        release(facade.device());
        ex::set_error(std::move(receiver), std::move(failure));
      }
    };

    template<class Receiver> [[nodiscard]] auto connect(Receiver receiver) && -> op_state<Receiver>
    {
      return op_state<Receiver>{
        .scopes = std::move(scopes),
        .receiver = std::move(receiver),
        .semaphores = {},
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
    std::optional<graph_presentation> presentation;
    Receiver receiver;
    using child_receiver_t = pass_graph_after_gpu_receiver<StepStorage, Receiver>;
    using fence_sender_t = detail::graph_submit_sender;
    using completion_sender_t = decltype(ex::continues_on(std::declval<fence_sender_t>(), std::declval<scheduler>()));
    using submit_op_t = decltype(ex::connect(std::declval<completion_sender_t>(), std::declval<child_receiver_t>()));

    struct submit_op_holder
    {
      submit_op_t op;
      submit_op_holder(detail::graph_submit_sender sender, scheduler sched, child_receiver_t child)
        : op(ex::connect(ex::continues_on(std::move(sender), std::move(sched)), std::move(child)))
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
      ex::set_error(std::move(receiver), std::move(failure));
    }

    auto start() noexcept -> void
    {
      if (detail::receiver_stop_requested(receiver)) {
        if (presentation) {
          if (auto abandoned = presentation->abort(presentation_abort_state::acquired_only); !abandoned) {
            ex::set_error(std::move(receiver), std::move(abandoned.error()));
            return;
          }
        }
        ex::set_stopped(std::move(receiver));
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
        auto planned = detail::plan_execution(detail::pass_step_count(steps), step_queues, facade.compute_queue_ref());
        if (!planned) {
          fail_before_submit(std::move(planned.error()));
          return;
        }
        if (presentation) {
          if (planned->batches.empty() || !detail::same_queue(planned->batches.back().queue, presentation->queue)) {
            fail_before_submit(make_error(
              errc::invalid_argument, "presentation requires a final batch on the presenter's graphics queue"));
            return;
          }
          auto sync = presentation->prepare();
          if (!sync) {
            error failure = std::move(sync.error());
            (void)presentation->abort(presentation_abort_state::acquired_only);
            ex::set_error(std::move(receiver), std::move(failure));
            return;
          }
          presentation->sync = expected_take(sync);
          if (presentation->sync.fence == VK_NULL_HANDLE
              || presentation->sync.image_available_wait.semaphore == VK_NULL_HANDLE
              || presentation->sync.render_finished_signal.semaphore == VK_NULL_HANDLE) {
            fail_before_submit(make_error(errc::invalid_argument, "presentation returned incomplete synchronization"));
            return;
          }
        }
        auto prepared =
          detail::record_execution(state, *planned, steps, presentation ? presentation->image : VK_NULL_HANDLE);
        if (!prepared) {
          fail_before_submit(std::move(prepared.error()));
          return;
        }
        auto &child = submit_op.emplace(
          detail::graph_submit_sender{ .scopes = expected_take(prepared), .presentation = std::move(presentation) },
          scheduler_access::make(state),
          child_receiver_t{ .rcvr = &receiver, .steps = &steps });
        ex::start(child.op);
#if VKEXEC_HAS_EXCEPTIONS
      } catch (...) {
        fail_before_submit(unexpected_exception_error());
      }
#endif
    }
  };

}// namespace detail

/**
 * Sender that records a list of pass steps and submits them for GPU execution.
 *
 * `start()` records and submits the graph, then returns without waiting for GPU
 * completion on the successful path. After the submission fence signals,
 * completion is transferred to the context host scheduler, where `after_gpu`
 * callbacks run and the receiver is completed.
 *
 * Built by piping `schedule()` into `compute_pass(...)` and optional barriers.
 *
 * ~~~~~~~~~~~{.cpp}
 * auto graph = ex::schedule(ctx->get_scheduler())
 *   | vkexec::compute_pass(*bound.pipe, bound.set, params, 10000)
 *   | vkexec::barrier::compute_to_compute()
 *   | vkexec::compute_pass(*bound.pipe, bound.set, params, 10000);
 * vkexec::sync_wait(std::move(graph));
 * ~~~~~~~~~~~
 *
 * @see compute_pass
 */
template<detail::static_pass_step... Steps> struct pass_graph_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures = pass_graph_completion_signatures;

  detail::context_handle state;
  std::tuple<Steps...> steps;
  std::vector<queue_affinity> step_queues;
  std::optional<graph_presentation> presentation;
  queue_affinity current_queue;

  using step_storage_t = std::tuple<Steps...>;

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .state = state }; }

  template<class Receiver> using op_state = detail::pass_graph_op_state<step_storage_t, Receiver>;

  template<class Receiver>
    requires(std::copy_constructible<Steps> && ...)
  [[nodiscard]] auto connect(Receiver receiver) const & noexcept(
    std::is_nothrow_copy_constructible_v<step_storage_t> && std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .state = state,
      .steps = steps,
      .step_queues = step_queues,
      .presentation = presentation,
      .receiver = std::move(receiver),
      .submit_op = std::nullopt,
    };
  }

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) && noexcept(
    std::is_nothrow_move_constructible_v<step_storage_t> && std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .state = std::move(state),
      .steps = std::move(steps),
      .step_queues = std::move(step_queues),
      .presentation = std::move(presentation),
      .receiver = std::move(receiver),
      .submit_op = std::nullopt,
    };
  }
};

struct dynamic_pass_graph_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures = pass_graph_completion_signatures;
  using step_storage_t = std::vector<detail::dynamic_pass_step>;

  detail::context_handle state;
  step_storage_t steps;
  std::vector<queue_affinity> step_queues;
  std::optional<graph_presentation> presentation;
  queue_affinity current_queue;

  dynamic_pass_graph_sender() = default;
  explicit dynamic_pass_graph_sender(detail::context_handle graph_state) : state(std::move(graph_state)) {}
  ~dynamic_pass_graph_sender() = default;
  dynamic_pass_graph_sender(dynamic_pass_graph_sender &&) noexcept = default;
  auto operator=(dynamic_pass_graph_sender &&) noexcept -> dynamic_pass_graph_sender & = default;
  dynamic_pass_graph_sender(dynamic_pass_graph_sender const &) = delete;
  auto operator=(dynamic_pass_graph_sender const &) -> dynamic_pass_graph_sender & = delete;

  auto reserve(std::size_t count) -> void { steps.reserve(count); }

  //! Sets the queue affinity of subsequently appended pass steps.
  auto append_queue(queue_ref queue) -> dynamic_pass_graph_sender &
  {
    current_queue = queue;
    return *this;
  }

  //! Lowers and appends one primitive semantic pass operation to this runtime graph.
  template<class Tag, class Data>
    requires requires(Tag tag, Data &&data, scheduler_env const &env) {
      lower_vkexec_pass_step(tag, std::move(data), env);
    }
  auto append(detail::expr_closure<Tag, Data> operation) -> dynamic_pass_graph_sender &;

  auto append(detail::expr_closure<detail::on_queue_t, queue_ref> operation) -> dynamic_pass_graph_sender &
  { return append_queue(operation.data); }

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .state = state }; }

  template<class Receiver> using op_state = detail::pass_graph_op_state<step_storage_t, Receiver>;

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) && noexcept(
    std::is_nothrow_move_constructible_v<step_storage_t> && std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .state = std::move(state),
      .steps = std::move(steps),
      .step_queues = std::move(step_queues),
      .presentation = std::move(presentation),
      .receiver = std::move(receiver),
      .submit_op = std::nullopt,
    };
  }
};

/**
 * Builds a direct-dispatch compute pass with push constants `params`.
 *
 * @param bind Pipeline / layout / set.
 * @param params Trivially copyable push-constant blob.
 * @param groups Workgroup counts.
 */
struct compute_pass_t
{
  template<detail::push_constant_type Params>
  [[nodiscard]] auto operator()(compute_bind bind, Params const &params, dispatch groups) const
    -> detail::expr_closure<compute_pass_t, detail::compute_pass_data<Params, dispatch>>;

  template<detail::push_constant_type Params>
  [[nodiscard]] auto operator()(compute_bind bind, Params const &params, indirect_dispatch groups) const
    -> detail::expr_closure<compute_pass_t, detail::compute_pass_data<Params, indirect_dispatch>>;

  [[nodiscard]] auto operator()(compute_bind bind, dispatch groups) const
    -> detail::expr_closure<compute_pass_t, detail::compute_pass_data<detail::no_push_constants, dispatch>>;
  [[nodiscard]] auto operator()(compute_bind bind, indirect_dispatch groups) const
    -> detail::expr_closure<compute_pass_t, detail::compute_pass_data<detail::no_push_constants, indirect_dispatch>>;

  template<detail::push_constant_type Params>
  [[nodiscard]] auto
    operator()(handles::compute_pipeline const &pipe, VkDescriptorSet set, Params const &params, dispatch groups) const;

  template<detail::push_constant_type Params>
  [[nodiscard]] auto operator()(handles::compute_pipeline const &pipe,
    VkDescriptorSet set,
    Params const &params,
    std::uint32_t work_count) const;

  [[nodiscard]] auto
    operator()(handles::compute_pipeline const &pipe, VkDescriptorSet set, std::uint32_t work_count) const;

  template<class Bound, detail::push_constant_type Params>
    requires requires(Bound const &bound, std::uint32_t count) {
      { bound.bind() } -> std::same_as<compute_bind>;
      { bound.pipe->groups_for(count) } -> std::same_as<dispatch>;
    }
  [[nodiscard]] auto operator()(Bound const &bound, Params const &params, std::uint32_t work_count) const
  { return (*this)(bound.bind(), params, bound.pipe->groups_for(work_count)); }

  template<class Bound>
    requires requires(Bound const &bound, std::uint32_t count) {
      { bound.bind() } -> std::same_as<compute_bind>;
      { bound.pipe->groups_for(count) } -> std::same_as<dispatch>;
    }
  [[nodiscard]] auto operator()(Bound const &bound, std::uint32_t work_count) const
  { return (*this)(bound.bind(), bound.pipe->groups_for(work_count)); }

  template<class Pipe, detail::push_constant_type Params>
    requires requires(Pipe const &pipe, VkDescriptorSet set, std::uint32_t count) {
      pipe.bind(set);
      pipe.groups_for(count);
    }
  [[nodiscard]] auto
    operator()(Pipe const &pipe, VkDescriptorSet set, Params const &params, std::uint32_t work_count) const
  { return (*this)(pipe.bind(set), params, pipe.groups_for(work_count)); }

  template<class Pipe>
    requires requires(Pipe const &pipe, VkDescriptorSet set, std::uint32_t count) {
      pipe.bind(set);
      pipe.groups_for(count);
    }
  [[nodiscard]] auto operator()(Pipe const &pipe, VkDescriptorSet set, std::uint32_t work_count) const
  { return (*this)(pipe.bind(set), pipe.groups_for(work_count)); }

  template<class... Args>
    requires requires(compute_pass_t const &self, Args &&...args) {
      compute_pass_custom(self, std::forward<Args>(args)...);
    }
  [[nodiscard]] auto operator()(Args &&...args) const
    -> decltype(compute_pass_custom(*this, std::forward<Args>(args)...))
  { return compute_pass_custom(*this, std::forward<Args>(args)...); }

  template<vkexec_predecessor Sender, class... Args>
    requires requires(compute_pass_t const &self, Sender &&sender, Args &&...args) {
      std::forward<Sender>(sender) | self(std::forward<Args>(args)...);
    }
  [[nodiscard]] auto operator()(Sender &&sender, Args &&...args) const
    noexcept(noexcept(std::forward<Sender>(sender) | (*this)(std::forward<Args>(args)...)))
      -> decltype(std::forward<Sender>(sender) | (*this)(std::forward<Args>(args)...))
  { return std::forward<Sender>(sender) | (*this)(std::forward<Args>(args)...); }
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr compute_pass_t compute_pass{};

//! Returns the specialized local workgroup size as a `dispatch`.
[[nodiscard]] inline auto local_size(handles::compute_pipeline const &pipe) noexcept -> dispatch
{
  return dispatch{
    .x = std::get<0>(pipe.local_size), .y = std::get<1>(pipe.local_size), .z = std::get<2>(pipe.local_size)
  };
}

//! Returns workgroup counts covering `work_count` invocations along X.
[[nodiscard]] inline auto groups_for(handles::compute_pipeline const &pipe, std::uint32_t work_count) noexcept
  -> dispatch
{ return dispatch_groups_for(work_count, std::get<0>(pipe.local_size)); }

//! Convenience overload that binds `pipe` with `set` before building the closure.
namespace detail {

  template<class T> struct is_pass_graph_sender : std::false_type
  {
  };

  template<class... Steps> struct is_pass_graph_sender<pass_graph_sender<Steps...>> : std::true_type
  {
  };

  template<> struct is_pass_graph_sender<dynamic_pass_graph_sender> : std::true_type
  {
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  template<class T> inline constexpr bool is_pass_graph_sender_v = is_pass_graph_sender<std::remove_cvref_t<T>>::value;

  struct raw_pass_step_t
  {
  };


  template<static_pass_step Step> struct raw_pass_step_data
  {
    VKEXEC_NO_UNIQUE_ADDRESS Step step;
  };

}// namespace detail

//! Changes the queue affinity of subsequent passes in a graph.
[[nodiscard]] inline auto on_queue(queue_ref queue) -> detail::expr_closure<detail::on_queue_t, queue_ref>
{ return detail::make_expr_closure(detail::on_queue_t{}, queue); }

/**
 * Records raw Vulkan commands as a graph step. The graph owns command-buffer
 * creation, recording boundaries, submission, and completion. `resources`
 * describes accesses made by the recorder, including their buffer ranges or
 * image subresources. The graph uses these declarations to insert layout
 * transitions, memory barriers, and queue-family ownership transfers. The
 * first image read must declare its known initial layout. Manual barriers
 * inside the recorder must not change a declared resource's tracked state.
 * Cross-family transfers assume exclusive sharing. The first queue family
 * using a resource is assumed to own it already; imported resources must be
 * synchronized externally before their first graph use.
 *
 * The recorder accepts a recording VkCommandBuffer and returns void or status.
 */
template<resource_use... Uses, class Record>
  requires std::move_constructible<std::decay_t<Record>> && std::invocable<std::decay_t<Record> &, VkCommandBuffer>
           && (std::same_as<std::invoke_result_t<std::decay_t<Record> &, VkCommandBuffer>, void>
               || std::same_as<std::invoke_result_t<std::decay_t<Record> &, VkCommandBuffer>, status>)
[[nodiscard]] auto custom_pass(resource_uses<Uses...> resources, Record &&record)
  -> detail::expr_closure<detail::custom_pass_t, detail::custom_pass_data<resource_uses<Uses...>, std::decay_t<Record>>>
{
  return detail::make_expr_closure(detail::custom_pass_t{},
    detail::custom_pass_data<resource_uses<Uses...>, std::decay_t<Record>>{
      .resources = std::move(resources), .record_fn = std::forward<Record>(record) });
}

template<detail::static_pass_step Step>
[[nodiscard]] auto make_pass_adaptor(Step step) noexcept(std::is_nothrow_move_constructible_v<Step>)
  -> detail::expr_closure<detail::raw_pass_step_t, detail::raw_pass_step_data<Step>>
{
  return detail::make_expr_closure(
    detail::raw_pass_step_t{}, detail::raw_pass_step_data<Step>{ .step = std::move(step) });
}

[[nodiscard]] inline auto make_dynamic_pass_graph(schedule_sender snd) -> dynamic_pass_graph_sender
{ return dynamic_pass_graph_sender{ std::move(snd.state) }; }

template<detail::push_constant_type Params>
// NOLINTNEXTLINE(performance-unnecessary-value-param)
auto compute_pass_t::operator()(compute_bind bind, Params const &params, dispatch groups) const
  -> detail::expr_closure<compute_pass_t, detail::compute_pass_data<Params, dispatch>>
{
  return detail::make_expr_closure(*this,
    detail::compute_pass_data<Params, dispatch>{ .bind = std::move(bind), .push = params, .dispatch_info = groups });
}

template<detail::push_constant_type Params>
// NOLINTNEXTLINE(performance-unnecessary-value-param)
auto compute_pass_t::operator()(compute_bind bind, Params const &params, indirect_dispatch groups) const
  -> detail::expr_closure<compute_pass_t, detail::compute_pass_data<Params, indirect_dispatch>>
{
  return detail::make_expr_closure(*this,
    detail::compute_pass_data<Params, indirect_dispatch>{
      .bind = std::move(bind), .push = params, .dispatch_info = groups });
}

inline auto compute_pass_t::operator()(compute_bind bind, dispatch groups) const
  -> detail::expr_closure<compute_pass_t, detail::compute_pass_data<detail::no_push_constants, dispatch>>
{
  return detail::make_expr_closure(*this,
    detail::compute_pass_data<detail::no_push_constants, dispatch>{
      .bind = std::move(bind), .push = {}, .dispatch_info = groups });
}

inline auto compute_pass_t::operator()(compute_bind bind, indirect_dispatch groups) const
  -> detail::expr_closure<compute_pass_t, detail::compute_pass_data<detail::no_push_constants, indirect_dispatch>>
{
  return detail::make_expr_closure(*this,
    detail::compute_pass_data<detail::no_push_constants, indirect_dispatch>{
      .bind = std::move(bind), .push = {}, .dispatch_info = groups });
}

template<detail::push_constant_type Params>
auto compute_pass_t::operator()(handles::compute_pipeline const &pipe,
  VkDescriptorSet set,
  Params const &params,
  dispatch groups) const
{ return (*this)(bind_compute(pipe, set), params, groups); }

template<detail::push_constant_type Params>
auto compute_pass_t::operator()(handles::compute_pipeline const &pipe,
  VkDescriptorSet set,
  Params const &params,
  std::uint32_t work_count) const
{ return (*this)(pipe, set, params, groups_for(pipe, work_count)); }

inline auto
  compute_pass_t::operator()(handles::compute_pipeline const &pipe, VkDescriptorSet set, std::uint32_t work_count) const
{ return (*this)(bind_compute(pipe, set), groups_for(pipe, work_count)); }

}// namespace vkexec

#include <vkexec/detail/pass_lowering.hpp>

namespace vkexec {

template<class Tag, class Data>
  requires requires(Tag tag, Data &&data, scheduler_env const &env) {
    lower_vkexec_pass_step(tag, std::move(data), env);
  }
auto dynamic_pass_graph_sender::append(detail::expr_closure<Tag, Data> operation) -> dynamic_pass_graph_sender &
{
  assert(step_queues.empty() || step_queues.size() == steps.size());
  if (step_queues.empty()) { step_queues.resize(steps.size()); }
  steps.emplace_back(lower_vkexec_pass_step(std::move(operation.tag), std::move(operation.data), get_env()));
  step_queues.push_back(current_queue);
  return *this;
}


}// namespace vkexec

#endif// VKEXEC_PASS_HPP
