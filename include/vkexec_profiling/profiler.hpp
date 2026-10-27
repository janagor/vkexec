#ifndef VKEXEC_PROFILING_PROFILER_HPP
#define VKEXEC_PROFILING_PROFILER_HPP

#include <vkexec/context.hpp>
#include <vkexec/result.hpp>
#include <vkexec_profiling/profile_event.hpp>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace vkexec::profiling {

struct profiler_options
{
  static constexpr std::uint32_t k_default_queries_per_pool = 512;
  static constexpr std::uint32_t k_default_max_in_flight_regions = 8192;
  static constexpr std::uint32_t k_default_max_completed_events = 65536;

  std::uint32_t queries_per_pool{ k_default_queries_per_pool };
  std::uint32_t max_in_flight_regions{ k_default_max_in_flight_regions };
  std::uint32_t max_completed_events{ k_default_max_completed_events };
};

namespace detail {
  class profiler_state;
  template<class Step> class profiled_pass_step;
}// namespace detail
class manual_capture;

class profiler
{
public:
  [[nodiscard]] static auto create(context &ctx, profiler_options options = {}) -> result<profiler>;

  [[nodiscard]] auto take_completed() -> std::vector<profile_event>;
  [[nodiscard]] auto supported(queue_ref queue) const noexcept -> bool;
  [[nodiscard]] auto dropped_events() const noexcept -> std::uint64_t;

private:
  friend class manual_capture;
  template<class Step> friend class detail::profiled_pass_step;
  explicit profiler(std::shared_ptr<detail::profiler_state> state) : state_(std::move(state)) {}
  std::shared_ptr<detail::profiler_state> state_;
};

}// namespace vkexec::profiling

#endif// VKEXEC_PROFILING_PROFILER_HPP
