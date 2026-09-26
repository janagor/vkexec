#ifndef VKEXEC_PASS_HPP
#define VKEXEC_PASS_HPP

//! \file
//! Compute pass recording, pass graphs, and stdexec pipe adaptors.

#include <vkexec/error.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/push.hpp>
#include <vkexec/result.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit_scope.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

namespace ex = stdexec;

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
 * Y and Z remain 1. A zero `local_x` is treated as 1.
 */
[[nodiscard]] constexpr auto dispatch_groups_for(std::uint32_t work_count, std::uint32_t local_x) noexcept -> dispatch
{
  std::uint32_t const group_size = local_x == 0U ? 1U : local_x;
  return dispatch{ .x = (work_count + group_size - 1U) / group_size };
}

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
};

//! Builds a `compute_bind` from pipeline resources and an optional set.
[[nodiscard]] auto bind_compute(handles::compute_pipeline const &pipe, VkDescriptorSet set = VK_NULL_HANDLE)
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
auto record_pass(VkCommandBuffer cmd, compute_bind bind, void const *push, std::uint32_t push_bytes, dispatch groups)
  -> void;

/**
 * Records bind, optional push constants, and an indirect dispatch on `cmd`.
 *
 * @param groups Buffer and offset containing `VkDispatchIndirectCommand`.
 */
auto record_pass(VkCommandBuffer cmd,
  compute_bind bind,
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
    };

    template<static_pass_step Step> struct model final : interface
    {
      Step step;

      explicit model(Step value) : step(std::move(value)) {}

      auto record(context &ctx, VkCommandBuffer cmd, pass_cleanup &cleanup) -> status override
      { return step.record(ctx, cmd, cleanup); }

      auto after_gpu() -> void override { detail::run_after_gpu(step); }
    };

    std::unique_ptr<interface> impl_;

  public:
    dynamic_pass_step() = delete;

    template<static_pass_step Step>
      requires(!std::same_as<std::remove_cvref_t<Step>, dynamic_pass_step>)
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
  };

  [[nodiscard]] inline auto push_bytes(no_push_constants const & /*unused*/) noexcept -> std::span<std::byte const>
  { return {}; }

  template<class Push>
    requires std::is_trivially_copyable_v<Push>
  [[nodiscard]] auto push_bytes(Push const &push) noexcept -> std::span<std::byte const>
  { return std::as_bytes(std::span{ &push, std::size_t{ 1 } }); }

  template<push_constant_type Push, dispatch_kind Dispatch> struct compute_pass_step
  {
    compute_bind bind{};
    [[no_unique_address]] Push push{};
    Dispatch dispatch_info{};

    auto record(context & /*ctx*/, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> status
    {
      auto const bytes = push_bytes(push);
      void const *const data = bytes.empty() ? nullptr : static_cast<void const *>(bytes.data());
      record_pass(cmd, bind, data, static_cast<std::uint32_t>(bytes.size()), dispatch_info);
      return {};
    }
  };

  template<push_constant_type Push, dispatch_kind Dispatch>
  [[nodiscard]] auto make_compute_pass_step(compute_bind bind, Push const &push, Dispatch dispatch_info)
    -> compute_pass_step<Push, Dispatch>
  { return { .bind = bind, .push = push, .dispatch_info = dispatch_info }; }

  template<class Tag> struct barrier_step
  {
    [[no_unique_address]] Tag tag;

    auto record(context & /*ctx*/, VkCommandBuffer cmd, pass_cleanup & /*cleanup*/) -> status
    {
      tag(cmd);
      return {};
    }
  };

  template<class Tag> [[nodiscard]] auto make_barrier_step(Tag tag) -> barrier_step<Tag>
  { return barrier_step<Tag>{ .tag = std::move(tag) }; }

  template<class Record> struct callback_pass_step
  {
    [[no_unique_address]] Record record_fn;

    auto record(context &ctx, VkCommandBuffer cmd, pass_cleanup &cleanup) -> status
    { return record_fn(ctx, cmd, cleanup); }
  };

  template<class Record, class AfterGpu> struct callback_after_gpu_pass_step
  {
    [[no_unique_address]] Record record_fn;
    [[no_unique_address]] AfterGpu after_gpu_fn;

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
    if (auto recorded = record_static_steps(*scope.ctx, scope.cmd, scope.cleanup, steps); !recorded) {
      return fail(recorded);
    }
    return scope.end_recording();
  }

  [[nodiscard]] inline auto record_pass_steps(submit_scope &scope, std::vector<dynamic_pass_step> &steps) -> status
  {
    if (auto recorded = record_dynamic_steps(*scope.ctx, scope.cmd, scope.cleanup, steps); !recorded) {
      return fail(recorded);
    }
    return scope.end_recording();
  }

  template<class StepStorage>
  [[nodiscard]] auto open_and_record_pass(context *ctx, StepStorage &steps) -> result<submit_scope>
  {
    auto opened = submit_scope::open(*ctx);
    if (!opened) { return fail(opened); }
    submit_scope scope = expected_take(opened);
    if (auto recorded = record_pass_steps(scope, steps); !recorded) {
      scope.release();
      return fail(recorded);
    }
    return scope;
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
#if VKEXEC_ENABLE_EXCEPTIONS
      try {
#endif
        detail::run_after_gpu(*steps);
#if VKEXEC_ENABLE_EXCEPTIONS
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

  template<class StepStorage, class Receiver> struct pass_graph_op_state
  {
    context *ctx{ nullptr };
    StepStorage steps;
    Receiver receiver;
    using child_receiver_t = pass_graph_after_gpu_receiver<StepStorage, Receiver>;
    using fence_sender_t = detail::submit_fence_sender;
    using completion_sender_t = decltype(ex::continues_on(std::declval<fence_sender_t>(), std::declval<scheduler>()));
    using submit_op_t = decltype(ex::connect(std::declval<completion_sender_t>(), std::declval<child_receiver_t>()));

    struct submit_op_holder
    {
      submit_op_t op;
      submit_op_holder(detail::submit_fence_sender sender, scheduler sched, child_receiver_t child)
        : op(ex::connect(ex::continues_on(std::move(sender), sched), std::move(child)))
      {}
      ~submit_op_holder() = default;
      submit_op_holder(submit_op_holder const &) = delete;
      auto operator=(submit_op_holder const &) -> submit_op_holder & = delete;
      submit_op_holder(submit_op_holder &&) = delete;
      auto operator=(submit_op_holder &&) -> submit_op_holder & = delete;
    };

    std::optional<submit_op_holder> submit_op;

    auto start() noexcept -> void
    {
      auto const token = ex::get_stop_token(ex::get_env(receiver));
      if constexpr (!ex::unstoppable_token<std::remove_cvref_t<decltype(token)>>) {
        if (token.stop_requested()) {
          ex::set_stopped(std::move(receiver));
          return;
        }
      }
#if VKEXEC_ENABLE_EXCEPTIONS
      try {
#endif
        auto prepared = detail::open_and_record_pass(ctx, steps);
        if (!prepared) {
          ex::set_error(std::move(receiver), std::move(prepared.error()));
          return;
        }
        auto &child = submit_op.emplace(detail::submit_fence(expected_take(prepared)),
          ctx->get_scheduler(),
          child_receiver_t{ .rcvr = &receiver, .steps = &steps });
        ex::start(child.op);
#if VKEXEC_ENABLE_EXCEPTIONS
      } catch (...) {
        ex::set_error(std::move(receiver), unexpected_exception_error());
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

  context *ctx{ nullptr };
  std::tuple<Steps...> steps;

  using step_storage_t = std::tuple<Steps...>;

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  template<class Receiver> using op_state = detail::pass_graph_op_state<step_storage_t, Receiver>;

  template<class Receiver>
    requires(std::copy_constructible<Steps> && ...)
  [[nodiscard]] auto connect(Receiver receiver) const & noexcept(
    std::is_nothrow_copy_constructible_v<step_storage_t> && std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = ctx,
      .steps = steps,
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
      .ctx = ctx,
      .steps = std::move(steps),
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

  context *ctx{ nullptr };
  step_storage_t steps;

  dynamic_pass_graph_sender() = default;
  explicit dynamic_pass_graph_sender(context *graph_ctx) : ctx(graph_ctx) {}
  ~dynamic_pass_graph_sender() = default;
  dynamic_pass_graph_sender(dynamic_pass_graph_sender &&) noexcept = default;
  auto operator=(dynamic_pass_graph_sender &&) noexcept -> dynamic_pass_graph_sender & = default;
  dynamic_pass_graph_sender(dynamic_pass_graph_sender const &) = delete;
  auto operator=(dynamic_pass_graph_sender const &) -> dynamic_pass_graph_sender & = delete;

  auto reserve(std::size_t count) -> void { steps.reserve(count); }

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .ctx = ctx }; }

  template<class Receiver> using op_state = detail::pass_graph_op_state<step_storage_t, Receiver>;

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) && noexcept(
    std::is_nothrow_move_constructible_v<step_storage_t> && std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .ctx = ctx,
      .steps = std::move(steps),
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
  [[nodiscard]] auto operator()(compute_bind bind, Params const &params, dispatch groups) const;

  template<detail::push_constant_type Params>
  [[nodiscard]] auto operator()(compute_bind bind, Params const &params, indirect_dispatch groups) const;

  [[nodiscard]] auto operator()(compute_bind bind, dispatch groups) const;
  [[nodiscard]] auto operator()(compute_bind bind, indirect_dispatch groups) const;

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

/**
 * Builds an indirect-dispatch compute pass with push constants `params`.
 *
 * @param groups Buffer containing `VkDispatchIndirectCommand` at `offset`.
 */

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

  template<static_pass_step Step>
  [[nodiscard]] auto make_pass_graph(context *ctx, Step step) noexcept(std::is_nothrow_move_constructible_v<Step>)
    -> pass_graph_sender<Step>
  { return pass_graph_sender<Step>{ .ctx = ctx, .steps = std::tuple<Step>{ std::move(step) } }; }

  template<class... Steps, static_pass_step NewStep>
  [[nodiscard]] auto append_step(pass_graph_sender<Steps...> graph, NewStep step) noexcept(
    (std::is_nothrow_move_constructible_v<Steps> && ...) && std::is_nothrow_move_constructible_v<NewStep>)
    -> pass_graph_sender<Steps..., NewStep>
  {
    context *const ctx = graph.ctx;
    return std::apply(
      [ctx, step = std::move(step)](auto &&...old) mutable noexcept(
        (std::is_nothrow_move_constructible_v<Steps> &&...) &&std::is_nothrow_move_constructible_v<NewStep>)
        -> pass_graph_sender<Steps..., NewStep> {
        return pass_graph_sender<Steps..., NewStep>{
          .ctx = ctx,
          .steps = std::tuple<Steps..., NewStep>{ std::forward<decltype(old)>(old)..., std::move(step) },
        };
      },
      std::move(graph.steps));
  }

}// namespace detail

/**
 * Lazy adaptor: after `pred` completes, builds one typed `pass_graph_sender`.
 *
 * Lowered by the vkexec domain via `lower_vkexec_sender`.
 * `get_completion_scheduler<set_value_t>` is preserved from the predecessor. All
 * composed steps remain fused in one tuple, and the resulting graph completes
 * on that context's host scheduler.
 */
template<class Pred, detail::static_pass_step... Steps> struct deferred_pass_graph_sender
{
  using sender_concept = ex::sender_t;
  using completion_signatures = pass_graph_completion_signatures;

  Pred pred;
  std::tuple<Steps...> steps;

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env
  {
    scheduler const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(pred));
    return scheduler_env{ .ctx = sched.get_context() };
  }
};

template<class Pred, detail::static_pass_step... Steps, class Env>
[[nodiscard]] auto
  lower_vkexec_sender(ex::set_value_t /*tag*/, deferred_pass_graph_sender<Pred, Steps...> sndr, Env const & /*env*/)
{
  scheduler const sched = ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr.pred));
  // NOLINTNEXTLINE(misc-const-correctness)
  context *const ctx = sched.get_context();
  return ex::let_value(
    std::move(sndr.pred), [ctx, steps = std::move(sndr.steps)](auto &&...) mutable -> pass_graph_sender<Steps...> {
      return pass_graph_sender<Steps...>{ .ctx = ctx, .steps = std::move(steps) };
    });
}

namespace detail {

  template<class T> struct is_deferred_pass_graph_sender : std::false_type
  {
  };

  template<class Pred, static_pass_step... Steps>
  struct is_deferred_pass_graph_sender<deferred_pass_graph_sender<Pred, Steps...>> : std::true_type
  {
  };

  template<class T>
  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr bool is_deferred_pass_graph_sender_v = is_deferred_pass_graph_sender<std::remove_cvref_t<T>>::value;

  template<static_pass_step Step>
  [[nodiscard]] auto apply_pass_step(schedule_sender snd, Step step) noexcept(
    noexcept(make_pass_graph(snd.ctx, std::move(step)))) -> pass_graph_sender<Step>
  { return detail::make_pass_graph(snd.ctx, std::move(step)); }

  template<class... Steps, static_pass_step Step>
  [[nodiscard]] auto apply_pass_step(pass_graph_sender<Steps...> graph, Step step) noexcept(
    noexcept(append_step(std::move(graph), std::move(step)))) -> pass_graph_sender<Steps..., Step>
  { return detail::append_step(std::move(graph), std::move(step)); }

  template<static_pass_step Step>
  [[nodiscard]] auto apply_pass_step(dynamic_pass_graph_sender graph, Step step) -> dynamic_pass_graph_sender
  {
    graph.steps.emplace_back(std::move(step));
    return graph;
  }

  /**
   * Wraps a vkexec predecessor in a lazy pass adaptor (domain-lowered later).
   *
   * Used when the left-hand side is not already a `schedule_sender` or `pass_graph_sender`.
   */
  template<class Pred, static_pass_step... Steps, static_pass_step Step>
  [[nodiscard]] auto apply_pass_step(deferred_pass_graph_sender<Pred, Steps...> sndr, Step step) noexcept(
    std::is_nothrow_move_constructible_v<Pred> && (std::is_nothrow_move_constructible_v<Steps> && ...)
    && std::is_nothrow_move_constructible_v<Step>) -> deferred_pass_graph_sender<Pred, Steps..., Step>
  {
    return std::apply(
      [&sndr, &step](Steps &&...old) noexcept(std::is_nothrow_move_constructible_v<Pred> &&(
        std::is_nothrow_move_constructible_v<Steps> &&...) &&std::is_nothrow_move_constructible_v<Step>)
        -> deferred_pass_graph_sender<Pred, Steps..., Step> {
        return { .pred = std::move(sndr.pred), .steps = { std::move(old)..., std::move(step) } };
      },
      std::move(sndr.steps));
  }

  template<vkexec_predecessor Pred, static_pass_step Step>
    requires(!std::same_as<std::remove_cvref_t<Pred>, schedule_sender> && !is_pass_graph_sender_v<Pred>
             && !is_deferred_pass_graph_sender_v<Pred>)
  [[nodiscard]] auto apply_pass_step(Pred &&pred, Step step) noexcept(
    std::is_nothrow_constructible_v<std::remove_cvref_t<Pred>, Pred> && std::is_nothrow_move_constructible_v<Step>)
    -> deferred_pass_graph_sender<std::remove_cvref_t<Pred>, Step>
  {
    return deferred_pass_graph_sender<std::remove_cvref_t<Pred>, Step>{ .pred = std::forward<Pred>(pred),
      .steps = { std::move(step) } };
  }

  template<static_pass_step Step> struct pass_step_closure : ex::sender_adaptor_closure<pass_step_closure<Step>>
  {
    [[no_unique_address]] Step step;

    template<class Sender>
      requires requires(Sender &&sender, Step &&value) {
        apply_pass_step(std::forward<Sender>(sender), std::move(value));
      }
    [[nodiscard]] auto operator()(Sender &&sender) && noexcept(
      noexcept(apply_pass_step(std::forward<Sender>(sender), std::move(step))))
      -> decltype(apply_pass_step(std::forward<Sender>(sender), std::move(step)))
    { return apply_pass_step(std::forward<Sender>(sender), std::move(step)); }

    template<class Sender>
      requires std::copy_constructible<Step>
               && requires(Sender &&sender, Step const &value) { apply_pass_step(std::forward<Sender>(sender), value); }
    [[nodiscard]] auto operator()(Sender &&sender) const & noexcept(
      noexcept(apply_pass_step(std::forward<Sender>(sender), step)))
      -> decltype(apply_pass_step(std::forward<Sender>(sender), step))
    { return apply_pass_step(std::forward<Sender>(sender), step); }
  };

}// namespace detail

template<detail::static_pass_step Step>
[[nodiscard]] auto make_pass_adaptor(Step step) noexcept(std::is_nothrow_move_constructible_v<Step>)
  -> detail::pass_step_closure<Step>
{ return detail::pass_step_closure<Step>{ {}, std::move(step) }; }

[[nodiscard]] inline auto make_dynamic_pass_graph(schedule_sender snd) -> dynamic_pass_graph_sender
{ return dynamic_pass_graph_sender{ snd.ctx }; }

template<detail::push_constant_type Params>
auto compute_pass_t::operator()(compute_bind bind, Params const &params, dispatch groups) const
{ return make_pass_adaptor(detail::make_compute_pass_step(bind, params, groups)); }

template<detail::push_constant_type Params>
auto compute_pass_t::operator()(compute_bind bind, Params const &params, indirect_dispatch groups) const
{ return make_pass_adaptor(detail::make_compute_pass_step(bind, params, groups)); }

inline auto compute_pass_t::operator()(compute_bind bind, dispatch groups) const
{ return make_pass_adaptor(detail::make_compute_pass_step(bind, detail::no_push_constants{}, groups)); }

inline auto compute_pass_t::operator()(compute_bind bind, indirect_dispatch groups) const
{ return make_pass_adaptor(detail::make_compute_pass_step(bind, detail::no_push_constants{}, groups)); }

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

#endif// VKEXEC_PASS_HPP
