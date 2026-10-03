#ifndef VKEXEC_BENCHMARK_REPRESENTATION_CANDIDATES_HPP
#define VKEXEC_BENCHMARK_REPRESENTATION_CANDIDATES_HPP

#include <vkexec/pass.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <memory_resource>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec_benchmark {

// Benchmark-only erasure: one operations pointer plus inline storage or a heap pointer.
class compact_step
{
  static constexpr std::size_t inline_bytes = 64;

  union storage {
    alignas(std::max_align_t) std::array<std::byte, inline_bytes> inline_data;
    void *heap;

    storage() noexcept : inline_data{} {}
    ~storage() {}
  } data_;

  struct operations
  {
    auto (*record)(storage &, vkexec::context &, VkCommandBuffer, vkexec::detail::pass_cleanup &) -> vkexec::status;
    auto (*after_gpu)(storage &) -> void;
    auto (*destroy)(storage &) noexcept -> void;
    auto (*move)(storage &, storage &) noexcept -> void;
  };

  operations const *ops_{ nullptr };

  template<class Step> [[nodiscard]] static auto inline_value(storage &value) noexcept -> Step *
  { return static_cast<Step *>(static_cast<void *>(value.inline_data.data())); }

  template<class Step>
  static constexpr operations inline_operations{
    .record = [](storage &value, vkexec::context &ctx, VkCommandBuffer cmd, vkexec::detail::pass_cleanup &cleanup)
      -> vkexec::status { return inline_value<Step>(value)->record(ctx, cmd, cleanup); },
    .after_gpu = [](storage &value) -> void { vkexec::detail::run_after_gpu(*inline_value<Step>(value)); },
    .destroy = [](storage &value) noexcept -> void { std::destroy_at(inline_value<Step>(value)); },
    .move = [](storage &dst, storage &src) noexcept -> void {
      std::construct_at(std::addressof(dst.inline_data));
      std::construct_at(inline_value<Step>(dst), std::move(*inline_value<Step>(src)));
      std::destroy_at(inline_value<Step>(src));
    },
  };

  template<class Step>
  static constexpr operations heap_operations{
    .record = [](storage &value, vkexec::context &ctx, VkCommandBuffer cmd, vkexec::detail::pass_cleanup &cleanup)
      -> vkexec::status { return static_cast<Step *>(value.heap)->record(ctx, cmd, cleanup); },
    .after_gpu = [](storage &value) -> void { vkexec::detail::run_after_gpu(*static_cast<Step *>(value.heap)); },
    .destroy = [](storage &value) noexcept -> void { delete static_cast<Step *>(value.heap); },
    .move = [](storage &dst, storage &src) noexcept -> void { dst.heap = std::exchange(src.heap, nullptr); },
  };

  auto release() noexcept -> void
  {
    if (ops_ != nullptr) { ops_->destroy(data_); }
    ops_ = nullptr;
  }

  auto take(compact_step &other) noexcept -> void
  {
    ops_ = other.ops_;
    if (ops_ != nullptr) { ops_->move(data_, other.data_); }
    other.ops_ = nullptr;
  }

public:
  compact_step() = delete;

  template<vkexec::detail::static_pass_step Step> explicit compact_step(Step step)
  {
    if constexpr (sizeof(Step) <= inline_bytes && alignof(Step) <= alignof(std::max_align_t)
                  && std::is_nothrow_move_constructible_v<Step>) {
      std::construct_at(inline_value<Step>(data_), std::move(step));
      ops_ = &inline_operations<Step>;
    } else {
      data_.heap = new Step(std::move(step));
      ops_ = &heap_operations<Step>;
    }
  }

  ~compact_step() { release(); }
  compact_step(compact_step &&other) noexcept { take(other); }
  auto operator=(compact_step &&other) noexcept -> compact_step &
  {
    if (this != &other) {
      release();
      take(other);
    }
    return *this;
  }
  compact_step(compact_step const &) = delete;
  auto operator=(compact_step const &) -> compact_step & = delete;

  auto record(vkexec::context &ctx, VkCommandBuffer cmd, vkexec::detail::pass_cleanup &cleanup) -> vkexec::status
  { return ops_->record(data_, ctx, cmd, cleanup); }

  auto after_gpu() -> void { ops_->after_gpu(data_); }
};

class counting_resource final : public std::pmr::memory_resource
{
  std::pmr::memory_resource *upstream_{ std::pmr::new_delete_resource() };

public:
  std::size_t allocations{};
  std::size_t bytes{};

private:
  auto do_allocate(std::size_t size, std::size_t alignment) -> void * override
  {
    void *ptr = upstream_->allocate(size, alignment);
    ++allocations;
    bytes += size;
    return ptr;
  }
  auto do_deallocate(void *ptr, std::size_t size, std::size_t alignment) -> void override
  { upstream_->deallocate(ptr, size, alignment); }
  [[nodiscard]] auto do_is_equal(std::pmr::memory_resource const &other) const noexcept -> bool override
  { return this == &other; }
};

class arena_graph
{
  struct interface
  {
    virtual ~interface() = default;
    virtual auto record(vkexec::context &, VkCommandBuffer, vkexec::detail::pass_cleanup &) -> vkexec::status = 0;
    virtual auto after_gpu() -> void = 0;
  };

  template<vkexec::detail::static_pass_step Step> struct model final : interface
  {
    Step step;
    explicit model(Step value) : step(std::move(value)) {}
    auto record(vkexec::context &ctx, VkCommandBuffer cmd, vkexec::detail::pass_cleanup &cleanup)
      -> vkexec::status override
    { return step.record(ctx, cmd, cleanup); }
    auto after_gpu() -> void override { vkexec::detail::run_after_gpu(step); }
  };

  counting_resource upstream_;
  std::pmr::monotonic_buffer_resource arena_{ &upstream_ };
  std::vector<interface *> steps_;

public:
  arena_graph() = default;
  arena_graph(arena_graph const &) = delete;
  auto operator=(arena_graph const &) -> arena_graph & = delete;
  arena_graph(arena_graph &&) = delete;
  auto operator=(arena_graph &&) -> arena_graph & = delete;
  ~arena_graph()
  {
    for (auto *step : steps_) { std::destroy_at(step); }
  }

  auto reserve(std::size_t count) -> void { steps_.reserve(count); }

  template<vkexec::detail::static_pass_step Step> auto append(Step step) -> void
  {
    void *slot = arena_.allocate(sizeof(model<Step>), alignof(model<Step>));
    auto *value = std::construct_at(static_cast<model<Step> *>(slot), std::move(step));
#if VKEXEC_HAS_EXCEPTIONS
    try {
      steps_.push_back(value);
    } catch (...) {
      std::destroy_at(value);
      throw;
    }
#else
    steps_.push_back(value);
#endif
  }

  auto record(vkexec::context &ctx, VkCommandBuffer cmd, vkexec::detail::pass_cleanup &cleanup) -> vkexec::status
  {
    for (auto *step : steps_) {
      auto recorded = step->record(ctx, cmd, cleanup);
      if (!recorded) { return vkexec::fail(recorded); }
    }
    return {};
  }

  [[nodiscard]] auto size() const noexcept -> std::size_t { return steps_.size(); }
  [[nodiscard]] auto upstream_allocations() const noexcept -> std::size_t { return upstream_.allocations; }
  [[nodiscard]] auto upstream_bytes() const noexcept -> std::size_t { return upstream_.bytes; }
  [[nodiscard]] static constexpr auto step_bytes() noexcept -> std::size_t { return sizeof(interface *); }
};

}// namespace vkexec_benchmark

#endif// VKEXEC_BENCHMARK_REPRESENTATION_CANDIDATES_HPP
