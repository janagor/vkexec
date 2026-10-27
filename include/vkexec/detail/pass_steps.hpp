#ifndef VKEXEC_DETAIL_PASS_STEPS_HPP
#define VKEXEC_DETAIL_PASS_STEPS_HPP

#include <vkexec/detail/pass_types.hpp>

namespace vkexec::detail {

  struct no_push_constants
  {
  };

  struct on_queue_t
  {
  };

  struct when_all_t
  {
  };

  inline constexpr std::size_t k_min_branch_count = 2;

  template<class... Branches> struct when_all_data
  {
    std::tuple<Branches...> branches;
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
  concept static_pass_step = std::move_constructible<Step>
                             && (requires(Step &step, context &ctx, VkCommandBuffer cmd, pass_cleanup &cleanup) {
                                  { step.record(ctx, cmd, cleanup) } -> std::same_as<status>;
                                }
                               || requires(Step &step,
                                 context &ctx,
                                 VkCommandBuffer cmd,
                                 pass_cleanup &cleanup,
                                 pass_recording_info const &info) {
                                    { step.record(ctx, cmd, cleanup, info) } -> std::same_as<status>;
                                  });

  template<static_pass_step Step>
  [[nodiscard]] auto
    record_step(Step &step, context &ctx, VkCommandBuffer cmd, pass_cleanup &cleanup, pass_recording_info const &info)
      -> status
  {
    if constexpr (requires {
                    { step.record(ctx, cmd, cleanup, info) } -> std::same_as<status>;
                  }) {
      return step.record(ctx, cmd, cleanup, info);
    } else {
      return step.record(ctx, cmd, cleanup);
    }
  }

  inline constexpr std::uint64_t k_first_submission_id = 1;
  [[nodiscard]] inline auto recording_info(queue_ref queue) noexcept -> pass_recording_info
  { return pass_recording_info{ .queue = queue }; }

  template<class T>
  concept sender_adaptor_closure =
    std::derived_from<std::decay_t<T>, stdexec::sender_adaptor_closure<std::decay_t<T>>>
    && std::move_constructible<std::decay_t<T>> && std::constructible_from<std::decay_t<T>, T>;

  template<class Step> auto run_after_gpu(Step &step) -> void
  {
    if constexpr (requires {
                    { step.after_gpu() } -> std::same_as<void>;
                  }) {
      step.after_gpu();
    }
  }

  template<class Step> auto commit_after_gpu(Step &step) -> void
  {
    if constexpr (requires {
                    { step.commit_after_gpu() } -> std::same_as<void>;
                  }) {
      step.commit_after_gpu();
    }
  }

  template<class Step> [[nodiscard]] auto step_has_resource_declarations(Step const &step) noexcept -> bool
  {
    if constexpr (requires { step.has_resource_declarations(); }) { return step.has_resource_declarations(); }
    if constexpr (requires { step.bind.metadata; }) { return step.bind.metadata == resource_metadata_state::complete; }
    return requires { &Step::resources; };
  }

  template<class Step> [[nodiscard]] auto step_requires_complete_metadata(Step const &step) noexcept -> bool
  {
    if constexpr (requires { step.requires_complete_metadata(); }) { return step.requires_complete_metadata(); }
    return requires { step.has_resource_declarations(); } || requires { sizeof(step.bind.metadata); };
  }

  template<class Step>
  [[nodiscard]] auto declare_step_resources(Step &step,
    resource_state_tracker &tracker,
    std::size_t index,
    queue_ref queue,
    bool crosses_families) -> status;

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

      virtual auto record(context &ctx, VkCommandBuffer cmd, pass_cleanup &cleanup, pass_recording_info const &info)
        -> status = 0;
      virtual auto after_gpu() -> void = 0;
      virtual auto commit_after_gpu() -> void = 0;
      virtual auto
        declare_resources(resource_state_tracker &tracker, std::size_t step, queue_ref queue, bool crosses_families)
          -> status = 0;
      [[nodiscard]] virtual auto has_resource_declarations() const noexcept -> bool = 0;
      [[nodiscard]] virtual auto requires_complete_metadata() const noexcept -> bool = 0;
    };

    template<static_pass_step Step> struct model final : interface
    {
      Step step;

      explicit model(Step value) : step(std::move(value)) {}

      auto record(context &ctx, VkCommandBuffer cmd, pass_cleanup &cleanup, pass_recording_info const &info)
        -> status override
      { return record_step(step, ctx, cmd, cleanup, info); }

      auto after_gpu() -> void override { detail::run_after_gpu(step); }

      auto commit_after_gpu() -> void override { detail::commit_after_gpu(step); }

      auto declare_resources(resource_state_tracker &tracker, std::size_t index, queue_ref queue, bool crosses_families)
        -> status override
      { return declare_step_resources(step, tracker, index, queue, crosses_families); }

      [[nodiscard]] auto has_resource_declarations() const noexcept -> bool override
      { return step_has_resource_declarations(step); }

      [[nodiscard]] auto requires_complete_metadata() const noexcept -> bool override
      { return step_requires_complete_metadata(step); }
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

    auto record(context &ctx, VkCommandBuffer cmd, pass_cleanup &cleanup, pass_recording_info const &info) -> status
    { return impl_->record(ctx, cmd, cleanup, info); }

    auto after_gpu() -> void { impl_->after_gpu(); }

    auto commit_after_gpu() -> void { impl_->commit_after_gpu(); }

    auto declare_resources(resource_state_tracker &tracker, std::size_t step, queue_ref queue, bool crosses_families)
      -> status
    { return impl_->declare_resources(tracker, step, queue, crosses_families); }

    [[nodiscard]] auto has_resource_declarations() const noexcept -> bool { return impl_->has_resource_declarations(); }

    [[nodiscard]] auto requires_complete_metadata() const noexcept -> bool
    { return impl_->requires_complete_metadata(); }
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

    [[nodiscard]] auto has_resource_declarations() const noexcept -> bool
    { return bind.metadata == resource_metadata_state::complete; }

    auto declare_resources(resource_state_tracker &tracker, std::size_t step, queue_ref queue) const -> status
    {
      if (!has_resource_declarations()) {
        return fail(errc::invalid_argument, "compute binding lacks resource or layout metadata");
      }
      if constexpr (std::same_as<Dispatch, indirect_dispatch>) {
        VKEXEC_TRY(tracker.use(
          read(dispatch_info.buffer, dispatch_info.offset, sizeof(VkDispatchIndirectCommand), buffer_usage::indirect),
          step,
          queue.family,
          queue.queue));
      }
      for (auto const &image : bind.images) { VKEXEC_TRY(tracker.use(image, step, queue.family, queue.queue)); }
      for (auto const &buffer : bind.buffers) { VKEXEC_TRY(tracker.use(buffer, step, queue.family, queue.queue)); }
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
  [[nodiscard]] auto record_static_steps(context &ctx,
    VkCommandBuffer cmd,
    pass_cleanup &cleanup,
    std::tuple<Steps...> &steps,
    pass_recording_info const &info) -> status
  {
    status result{};
    auto record_one = [&ctx, cmd, &cleanup, &info, &result](auto &step) -> bool {
      auto current = record_step(step, ctx, cmd, cleanup, info);
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
    std::vector<dynamic_pass_step> &steps,
    pass_recording_info const &info) -> status
  {
    for (auto &step : steps) {
      auto recorded = record_step(step, ctx, cmd, cleanup, info);
      if (!recorded) { return fail(recorded); }
    }
    return {};
  }

  template<class... Steps>
  [[nodiscard]] auto record_pass_steps(submit_scope &scope, std::tuple<Steps...> &steps) -> status
  {
    auto facade = context_access::facade(scope.state);
    if (auto recorded = record_static_steps(facade, scope.cmd, scope.cleanup, steps, recording_info(scope.queue));
      !recorded) {
      return fail(recorded);
    }
    return scope.end_recording();
  }

  [[nodiscard]] inline auto record_pass_steps(submit_scope &scope, std::vector<dynamic_pass_step> &steps) -> status
  {
    auto facade = context_access::facade(scope.state);
    if (auto recorded = record_dynamic_steps(facade, scope.cmd, scope.cleanup, steps, recording_info(scope.queue));
      !recorded) {
      return fail(recorded);
    }
    return scope.end_recording();
  }

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_PASS_STEPS_HPP
