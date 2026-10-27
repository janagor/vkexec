#ifndef VKEXEC_PROFILING_DETAIL_PROFILER_STATE_HPP
#define VKEXEC_PROFILING_DETAIL_PROFILER_STATE_HPP

#include <vkexec/context.hpp>
#include <vkexec/result.hpp>
#include <vkexec_profiling/profile_event.hpp>
#include <vkexec_profiling/profiler.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vkexec::profiling::detail {

struct query_pool_page
{
  query_pool_page() = default;
  query_pool_page(query_pool_page const &) = delete;
  auto operator=(query_pool_page const &) -> query_pool_page & = delete;
  query_pool_page(query_pool_page &&) = delete;
  auto operator=(query_pool_page &&) -> query_pool_page & = delete;
  ~query_pool_page();

  VkDevice device{ VK_NULL_HANDLE };
  VkQueryPool pool{ VK_NULL_HANDLE };
  std::uint32_t family{ 0 };
  std::vector<std::uint32_t> free_pairs;
};

class profiler_state;

class query_allocation
{
public:
  query_allocation() = default;
  query_allocation(std::shared_ptr<profiler_state> owner,
    std::shared_ptr<query_pool_page> pool_page,
    std::uint32_t query) noexcept;
  query_allocation(query_allocation const &) = delete;
  auto operator=(query_allocation const &) -> query_allocation & = delete;
  query_allocation(query_allocation &&other) noexcept;
  auto operator=(query_allocation &&other) noexcept -> query_allocation &;
  ~query_allocation();

  auto release() noexcept -> void;
  auto quarantine() noexcept -> void;

  std::shared_ptr<query_pool_page> page;
  std::uint32_t first_query{ 0 };

private:
  std::shared_ptr<profiler_state> owner_;
};

class profiler_state : public std::enable_shared_from_this<profiler_state>
{
public:
  profiler_state(context &ctx, profiler_options options, std::vector<VkQueueFamilyProperties> families, float period);
  profiler_state(profiler_state const &) = delete;
  auto operator=(profiler_state const &) -> profiler_state & = delete;
  profiler_state(profiler_state &&) = delete;
  auto operator=(profiler_state &&) -> profiler_state & = delete;
  ~profiler_state() = default;

  [[nodiscard]] auto supported(queue_ref queue) const noexcept -> bool;
  [[nodiscard]] auto reserve(queue_ref queue) -> result<query_allocation>;
  [[nodiscard]] auto resolve(query_allocation const &allocation,
    std::string name,
    std::uint64_t submission_id,
    std::uint32_t region_index) -> profile_event;
  auto publish(profile_event event) -> void;
  [[nodiscard]] auto take_completed() -> std::vector<profile_event>;
  [[nodiscard]] auto dropped_events() const noexcept -> std::uint64_t;
  [[nodiscard]] auto device() const noexcept -> VkDevice { return device_; }

private:
  friend class query_allocation;
  auto release_slot(std::shared_ptr<query_pool_page> const &page, std::uint32_t first_query) noexcept -> void;
  vkexec::detail::context_handle runtime_;
  VkDevice device_{ VK_NULL_HANDLE };
  profiler_options options_;
  std::vector<VkQueueFamilyProperties> families_;
  float period_{ 0.0F };
  std::mutex mutex_;
  std::vector<std::shared_ptr<query_pool_page>> pages_;
  std::vector<profile_event> completed_;
  std::uint32_t in_flight_{ 0 };
  std::atomic<std::uint64_t> dropped_{ 0 };
};

}// namespace vkexec::profiling::detail

#endif// VKEXEC_PROFILING_DETAIL_PROFILER_STATE_HPP
