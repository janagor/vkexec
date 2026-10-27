#include <vkexec_profiling/profiler.hpp>
#include <vkexec_profiling/detail/profiler_state.hpp>
#include "detail/timestamp_math.hpp"

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>
#include <vkexec_profiling/profile_event.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace vkexec::profiling {

auto profiler::create(context &ctx, profiler_options options) -> result<profiler>
{
  constexpr std::uint32_t k_queries_per_region = 2;
  if (ctx.device() == VK_NULL_HANDLE || ctx.physical_device() == VK_NULL_HANDLE ||
      options.queries_per_pool < k_queries_per_region || options.queries_per_pool % k_queries_per_region != 0 ||
      options.max_in_flight_regions == 0 || options.max_completed_events == 0) {
    return fail(errc::invalid_argument, "invalid profiler context or options");
  }

  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(ctx.physical_device(), &properties);
  if (!std::isfinite(properties.limits.timestampPeriod) || properties.limits.timestampPeriod <= 0.0F) {
    return fail(errc::unsupported, "device has no valid timestamp period");
  }

  std::uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical_device(), &count, nullptr);
  if (count == 0) { return fail(errc::unsupported, "device has no queue families"); }
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical_device(), &count, families.data());
  families.resize(count);
  if (count == 0) { return fail(errc::unsupported, "device has no queue families"); }
  return profiler{ std::make_shared<detail::profiler_state>(ctx, options, std::move(families),
                                                            properties.limits.timestampPeriod) };
}

auto profiler::take_completed() -> std::vector<profile_event> { return state_->take_completed(); }
auto profiler::supported(queue_ref queue) const noexcept -> bool { return state_->supported(queue); }
auto profiler::dropped_events() const noexcept -> std::uint64_t { return state_->dropped_events(); }

}// namespace vkexec::profiling

namespace vkexec::profiling::detail {

query_pool_page::~query_pool_page()
{
  if (pool != VK_NULL_HANDLE) { vkDestroyQueryPool(device, pool, nullptr); }
}

query_allocation::query_allocation(std::shared_ptr<profiler_state> owner, std::shared_ptr<query_pool_page> pool_page,
                                   std::uint32_t query) noexcept
  : page(std::move(pool_page)), first_query(query), owner_(std::move(owner))
{}

query_allocation::query_allocation(query_allocation &&other) noexcept
  : page(std::move(other.page)), first_query(other.first_query), owner_(std::move(other.owner_))
{}

auto query_allocation::operator=(query_allocation &&other) noexcept -> query_allocation &
{
  if (this == &other) { return *this; }
  release();
  page = std::move(other.page);
  first_query = other.first_query;
  owner_ = std::move(other.owner_);
  return *this;
}

query_allocation::~query_allocation() { release(); }

auto query_allocation::release() noexcept -> void
{
  if (!owner_) { return; }
  owner_->release_slot(page, first_query);
  page.reset();
  owner_.reset();
}

auto query_allocation::quarantine() noexcept -> void
{
  page.reset();
  owner_.reset();
}

profiler_state::profiler_state(context &ctx, profiler_options options, std::vector<VkQueueFamilyProperties> families,
                               float period)
  : runtime_(vkexec::detail::context_access::state(ctx)), device_(ctx.device()), options_(options),
    families_(std::move(families)), period_(period)
{}

auto profiler_state::supported(queue_ref queue) const noexcept -> bool
{
  if (queue.queue == VK_NULL_HANDLE || queue.family >= families_.size()) { return false; }
  auto const &family = families_.at(queue.family);
  constexpr VkQueueFlags k_reset_queues = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
  return family.timestampValidBits != 0 && (family.queueFlags & k_reset_queues) != 0;
}

auto profiler_state::reserve(queue_ref queue) -> result<query_allocation>
{
  constexpr std::uint32_t k_queries_per_region = 2;
  if (!supported(queue)) { return fail(errc::unsupported, "queue family cannot record timestamp queries"); }
  auto owner = weak_from_this().lock();
  if (!owner) { return fail(errc::invalid_argument, "profiler state must have shared ownership"); }
  std::scoped_lock const lock(mutex_);
  if (in_flight_ >= options_.max_in_flight_regions) {
    return fail(errc::out_of_range, "profiling region capacity exceeded");
  }
  auto page = std::shared_ptr<query_pool_page>{};
  for (auto const &candidate : pages_) {
    if (candidate->family == queue.family && !candidate->free_pairs.empty()) {
      page = candidate;
      break;
    }
  }
  if (!page) {
    VkQueryPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = options_.queries_per_pool;
    page = std::make_shared<query_pool_page>();
    page->device = device_;
    page->family = queue.family;
    auto const created = vkCreateQueryPool(device_, &info, nullptr, &page->pool);
    if (created != VK_SUCCESS) { return fail(created, "vkCreateQueryPool"); }
    page->free_pairs.reserve(options_.queries_per_pool / k_queries_per_region);
    for (std::uint32_t query = 0; query < options_.queries_per_pool; query += k_queries_per_region) {
      page->free_pairs.push_back(query);
    }
    pages_.push_back(page);
  }
  auto const first = page->free_pairs.back();
  page->free_pairs.pop_back();
  ++in_flight_;
  return query_allocation{ std::move(owner), std::move(page), first };
}

auto profiler_state::release_slot(std::shared_ptr<query_pool_page> const &page, std::uint32_t first_query) noexcept -> void
{
  std::scoped_lock const lock(mutex_);
  page->free_pairs.push_back(first_query);
  --in_flight_;
}

auto profiler_state::resolve(query_allocation const &allocation, std::string name, std::uint64_t submission_id,
                             std::uint32_t region_index) -> profile_event
{
  constexpr std::uint32_t k_queries_per_region = 2;
  constexpr std::size_t k_start_query_index = 0;
  constexpr std::size_t k_end_query_index = 1;
  struct query_value { std::uint64_t timestamp; std::uint64_t available; };
  std::array<query_value, k_queries_per_region> values{};
  auto event = profile_event{ .name = std::move(name), .submission_id = submission_id,
                              .queue_family = allocation.page->family, .region_index = region_index };
  auto const read = vkGetQueryPoolResults(device_, allocation.page->pool, allocation.first_query, k_queries_per_region,
                                          sizeof(values), values.data(), sizeof(query_value),
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
  if (read != VK_SUCCESS && read != VK_NOT_READY) {
    event.status = sample_status::readback_error;
  } else if (read == VK_NOT_READY || values.at(k_start_query_index).available == 0 ||
             values.at(k_end_query_index).available == 0) {
    event.status = sample_status::unavailable;
  } else if (auto duration = duration_ns(values.at(k_start_query_index).timestamp, values.at(k_end_query_index).timestamp,
                                         families_.at(allocation.page->family).timestampValidBits, period_)) {
    event.duration_ns = *duration;
    event.status = sample_status::valid;
    event.start_tick = values.at(k_start_query_index).timestamp;
    event.timestamp_valid_bits = families_.at(allocation.page->family).timestampValidBits;
    event.timestamp_period_ns = static_cast<double>(period_);
  } else {
    event.status = sample_status::invalid_timestamp;
  }
  return event;
}

auto profiler_state::publish(profile_event event) -> void
{
  std::scoped_lock const lock(mutex_);
  if (completed_.size() >= options_.max_completed_events) {
    ++dropped_;
    return;
  }
  completed_.push_back(std::move(event));
}

auto profiler_state::take_completed() -> std::vector<profile_event>
{
  std::scoped_lock const lock(mutex_);
  auto events = std::vector<profile_event>{};
  events.swap(completed_);
  return events;
}

auto profiler_state::dropped_events() const noexcept -> std::uint64_t { return dropped_.load(); }

}// namespace vkexec::profiling::detail
