#ifndef VKEXEC_PASS_HPP
#define VKEXEC_PASS_HPP

//! \file
//! Compute pass recording, pass graphs, and stdexec pipe adaptors.

#include <vkexec/detail/pass_graph_submit.hpp>

namespace vkexec {

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
 * auto graph = stdexec::schedule(ctx->get_scheduler())
 *   | vkexec::compute_pass(bound, params, work_count)
 *   | vkexec::barrier::compute_to_compute()
 *   | vkexec::compute_pass(bound, params, work_count);
 * vkexec::sync_wait(std::move(graph));
 * ~~~~~~~~~~~
 *
 * @see compute_pass
 */
template<detail::static_pass_step... Steps> struct pass_graph_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures = pass_graph_completion_signatures;

  detail::context_handle state;
  std::tuple<Steps...> steps;
  std::vector<queue_affinity> step_queues;
  std::vector<std::vector<std::size_t>> step_predecessors;
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
      .step_predecessors = step_predecessors,
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
      .step_predecessors = std::move(step_predecessors),
      .presentation = std::move(presentation),
      .receiver = std::move(receiver),
      .submit_op = std::nullopt,
    };
  }
};

struct dynamic_pass_graph_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures = pass_graph_completion_signatures;
  using step_storage_t = std::vector<detail::dynamic_pass_step>;

  detail::context_handle state;
  step_storage_t steps;
  std::vector<queue_affinity> step_queues;
  std::vector<std::vector<std::size_t>> step_predecessors;
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

  //! Adds a node after explicit predecessors. An empty span starts a branch.
  template<class Tag, class Data>
    requires requires(Tag tag, Data &&data, scheduler_env const &env) {
      lower_vkexec_pass_step(tag, std::move(data), env);
    }
  [[nodiscard]] auto append_after(std::span<std::size_t const> predecessors, detail::expr_closure<Tag, Data> operation)
    -> std::size_t;

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
      .step_predecessors = std::move(step_predecessors),
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

  template<detail::push_constant_type Params>
  [[nodiscard]] auto operator()(handles::compute_pipeline const &pipe,
    VkDescriptorSet set,
    resource_table const &resources,
    Params const &params,
    std::uint32_t work_count) const -> detail::expr_closure<compute_pass_t, detail::compute_pass_data<Params, dispatch>>
  { return (*this)(bind_compute(pipe, set, resources), params, groups_for(pipe, work_count)); }

  [[nodiscard]] auto operator()(handles::compute_pipeline const &pipe,
    VkDescriptorSet set,
    resource_table const &resources,
    std::uint32_t work_count) const
    -> detail::expr_closure<compute_pass_t, detail::compute_pass_data<detail::no_push_constants, dispatch>>
  { return (*this)(bind_compute(pipe, set, resources), groups_for(pipe, work_count)); }

  template<detail::push_constant_type Params>
  [[nodiscard]] auto operator()(handles::compute_pipeline const &pipe,
    VkDescriptorSet set,
    std::vector<image_use> images,
    std::vector<buffer_use> buffers,
    Params const &params,
    std::uint32_t work_count) const -> detail::expr_closure<compute_pass_t, detail::compute_pass_data<Params, dispatch>>
  {
    return (*this)(bind_compute(pipe, set).with_resources(std::move(images), std::move(buffers)),
      params,
      groups_for(pipe, work_count));
  }

  [[nodiscard]] auto operator()(handles::compute_pipeline const &pipe,
    VkDescriptorSet set,
    std::vector<image_use> images,
    std::vector<buffer_use> buffers,
    std::uint32_t work_count) const
    -> detail::expr_closure<compute_pass_t, detail::compute_pass_data<detail::no_push_constants, dispatch>>
  {
    return (*this)(
      bind_compute(pipe, set).with_resources(std::move(images), std::move(buffers)), groups_for(pipe, work_count));
  }

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

  template<class Pipe, detail::push_constant_type Params>
    requires requires(Pipe const &pipe, VkDescriptorSet set, resource_table const &resources, std::uint32_t count) {
      pipe.bind(set, resources);
      pipe.groups_for(count);
    }
  [[nodiscard]] auto operator()(Pipe const &pipe,
    VkDescriptorSet set,
    resource_table const &resources,
    Params const &params,
    std::uint32_t work_count) const -> decltype((*this)(pipe.bind(set, resources), params, pipe.groups_for(work_count)))
  { return (*this)(pipe.bind(set, resources), params, pipe.groups_for(work_count)); }

  template<class Pipe>
    requires requires(Pipe const &pipe, VkDescriptorSet set, resource_table const &resources, std::uint32_t count) {
      pipe.bind(set, resources);
      pipe.groups_for(count);
    }
  [[nodiscard]] auto
    operator()(Pipe const &pipe, VkDescriptorSet set, resource_table const &resources, std::uint32_t work_count) const
    -> decltype((*this)(pipe.bind(set, resources), pipe.groups_for(work_count)))
  { return (*this)(pipe.bind(set, resources), pipe.groups_for(work_count)); }

  template<class Pipe, detail::push_constant_type Params>
    requires requires(Pipe const &pipe, VkDescriptorSet set, std::uint32_t count) {
      pipe.bind(set);
      pipe.groups_for(count);
    }
  [[nodiscard]] auto operator()(Pipe const &pipe,
    VkDescriptorSet set,
    std::vector<image_use> images,
    std::vector<buffer_use> buffers,
    Params const &params,
    std::uint32_t work_count) const
    -> decltype((
      *this)(pipe.bind(set).with_resources(std::move(images), std::move(buffers)), params, pipe.groups_for(work_count)))
  {
    return (*this)(
      pipe.bind(set).with_resources(std::move(images), std::move(buffers)), params, pipe.groups_for(work_count));
  }

  template<class Pipe>
    requires requires(Pipe const &pipe, VkDescriptorSet set, std::uint32_t count) {
      pipe.bind(set);
      pipe.groups_for(count);
    }
  [[nodiscard]] auto operator()(Pipe const &pipe,
    VkDescriptorSet set,
    std::vector<image_use> images,
    std::vector<buffer_use> buffers,
    std::uint32_t work_count) const
    -> decltype((
      *this)(pipe.bind(set).with_resources(std::move(images), std::move(buffers)), pipe.groups_for(work_count)))
  { return (*this)(pipe.bind(set).with_resources(std::move(images), std::move(buffers)), pipe.groups_for(work_count)); }

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

//! Appends two or more independent fragments after the same incoming frontier.
//! Each branch must contain at least one pass; presentation belongs after the branches.
template<detail::sender_adaptor_closure... Branches>
  requires(sizeof...(Branches) >= detail::k_min_branch_count)
// Branches are forwarded into the tuple below.
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
[[nodiscard]] auto when_all(Branches &&...branches)
  -> detail::expr_closure<detail::when_all_t, detail::when_all_data<std::decay_t<Branches>...>>
{
  return detail::make_expr_closure(detail::when_all_t{},
    detail::when_all_data<std::decay_t<Branches>...>{
      .branches = std::tuple<std::decay_t<Branches>...>{ std::forward<Branches>(branches)... } });
}

/**
 * Records raw Vulkan commands as a graph step. The graph owns command-buffer
 * creation, recording boundaries, submission, and completion. `resources`
 * describes accesses made by the recorder, including their buffer ranges or
 * image subresources. The graph uses these declarations to insert layout
 * transitions, memory barriers, and queue-family ownership transfers. The
 * first image read must declare its known initial layout. Manual barriers
 * inside the recorder must not change a declared resource's tracked state.
 * Cross-family transfers assume exclusive sharing unless extent metadata
 * declares concurrent sharing. The first queue family
 * using a resource is assumed to own it already; imported resources must be
 * synchronized externally before their first graph use.
 * Optional buffer_extent/image_extent declarations register bounds and sharing
 * mode before accesses in this pass. Register bounds before the first use when
 * mixing symbolic ends with finite ranges that may reach the resource end.
 *
 * The recorder accepts a recording VkCommandBuffer and returns void or status.
 */
template<resource_declaration... Uses, class Record>
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
  if (!step_predecessors.empty()) {
    std::vector<std::size_t> predecessors;
    if (!steps.empty()) { predecessors.push_back(steps.size() - std::size_t{ 1 }); }
    step_predecessors.push_back(std::move(predecessors));
  }
  steps.emplace_back(lower_vkexec_pass_step(std::move(operation.tag), std::move(operation.data), get_env()));
  step_queues.push_back(current_queue);
  return *this;
}

template<class Tag, class Data>
  requires requires(Tag tag, Data &&data, scheduler_env const &env) {
    lower_vkexec_pass_step(tag, std::move(data), env);
  }
auto dynamic_pass_graph_sender::append_after(std::span<std::size_t const> predecessors,
  detail::expr_closure<Tag, Data> operation) -> std::size_t
{
  if (step_predecessors.empty()) {
    step_predecessors.resize(steps.size());
    for (auto index = std::size_t{ 1 }; index < steps.size(); ++index) {
      step_predecessors.at(index).push_back(index - std::size_t{ 1 });
    }
  }
  auto const node = steps.size();
  steps.emplace_back(lower_vkexec_pass_step(std::move(operation.tag), std::move(operation.data), get_env()));
  step_queues.push_back(current_queue);
  step_predecessors.emplace_back(predecessors.begin(), predecessors.end());
  return node;
}


}// namespace vkexec

#endif// VKEXEC_PASS_HPP
