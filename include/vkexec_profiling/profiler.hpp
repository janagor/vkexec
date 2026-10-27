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
  std::uint32_t queries_per_pool{ 512 };
  std::uint32_t max_in_flight_regions{ 8192 };
  std::uint32_t max_completed_events{ 65536 };
};

namespace detail {
  class profiler_state;
}
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
  explicit profiler(std::shared_ptr<detail::profiler_state> state) : state_(std::move(state)) {}
  std::shared_ptr<detail::profiler_state> state_;
};

}// namespace vkexec::profiling

#endif// VKEXEC_PROFILING_PROFILER_HPP
