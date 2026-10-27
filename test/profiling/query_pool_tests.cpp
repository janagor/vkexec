#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec_profiling/detail/profiler_state.hpp>
#include <vkexec_profiling/profiler.hpp>

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

static_assert(!std::is_copy_constructible_v<vkexec::profiling::detail::query_allocation>);
static_assert(std::is_nothrow_move_constructible_v<vkexec::profiling::detail::query_allocation>);

TEST_CASE("query pairs stay exclusive until released", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  constexpr std::uint32_t k_queries_per_pool = 4;
  constexpr std::uint32_t k_max_regions = 2;
  constexpr std::uint32_t k_max_events = 2;
  auto state = std::make_shared<vkexec::profiling::detail::profiler_state>(
    *ctx, vkexec::profiling::profiler_options{ .queries_per_pool = k_queries_per_pool,
                                               .max_in_flight_regions = k_max_regions,
                                               .max_completed_events = k_max_events },
    [] (vkexec::context &device) -> std::vector<VkQueueFamilyProperties> {
      std::uint32_t count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(device.physical_device(), &count, nullptr);
      std::vector<VkQueueFamilyProperties> families(count);
      vkGetPhysicalDeviceQueueFamilyProperties(device.physical_device(), &count, families.data());
      return families;
    }(*ctx),
    1.0F);
  auto const queue = ctx->compute_queue_ref();
  if (!state->supported(queue)) { SKIP("compute queue has no timestamp support"); }
  auto first = state->reserve(queue);
  auto second = state->reserve(queue);
  REQUIRE(first);
  REQUIRE(second);
  REQUIRE(first->first_query != second->first_query);
  REQUIRE_FALSE(state->reserve(queue));
  auto const recycled_index = first->first_query;
  first->release();
  auto recycled = state->reserve(queue);
  REQUIRE(recycled);
  REQUIRE(recycled->first_query == recycled_index);
  second->release();
  recycled->release();
}

TEST_CASE("query lease retains its allocator", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  std::uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device(), &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device(), &count, families.data());
  auto state = std::make_shared<vkexec::profiling::detail::profiler_state>(*ctx, vkexec::profiling::profiler_options{},
                                                                           std::move(families), 1.0F);
  auto const queue = ctx->compute_queue_ref();
  if (!state->supported(queue)) { SKIP("compute queue has no timestamp support"); }
  auto lease = state->reserve(queue);
  REQUIRE(lease);
  auto retained = std::weak_ptr{ state };
  state.reset();
  REQUIRE_FALSE(retained.expired());
  lease->release();
  REQUIRE(retained.expired());
}

TEST_CASE("query pools expand without reusing in-flight pairs", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  std::uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device(), &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device(), &count, families.data());
  constexpr std::uint32_t k_queries_per_pool = 2;
  constexpr std::uint32_t k_regions = 3;
  auto state = std::make_shared<vkexec::profiling::detail::profiler_state>(
    *ctx, vkexec::profiling::profiler_options{ .queries_per_pool = k_queries_per_pool,
                                               .max_in_flight_regions = k_regions },
    std::move(families), 1.0F);
  auto const queue = ctx->compute_queue_ref();
  if (!state->supported(queue)) { SKIP("compute queue has no timestamp support"); }
  auto first = state->reserve(queue);
  auto second = state->reserve(queue);
  auto third = state->reserve(queue);
  REQUIRE(first);
  REQUIRE(second);
  REQUIRE(third);
  REQUIRE(first->page->pool != second->page->pool);
  REQUIRE(first->page->pool != third->page->pool);
  REQUIRE(second->page->pool != third->page->pool);
  auto const first_pool = first->page->pool;
  first->release();
  second->release();
  third->release();
  auto reused = state->reserve(queue);
  REQUIRE(reused);
  REQUIRE(reused->page->pool == first_pool);
}

TEST_CASE("transfer-only families cannot use command-buffer query reset", "[vkexec][profiling]")
{
  auto ctx = vkexec::test::require_context();
  VkQueueFamilyProperties transfer_family{};
  transfer_family.queueFlags = VK_QUEUE_TRANSFER_BIT;
  transfer_family.timestampValidBits = 64;
  auto state = vkexec::profiling::detail::profiler_state(*ctx, {}, { transfer_family }, 1.0F);
  auto const queue = vkexec::queue_ref{ .queue = ctx->compute_queue(), .family = 0 };
  REQUIRE_FALSE(state.supported(queue));
  REQUIRE_FALSE(state.reserve(queue));
}

TEST_CASE("completed-event capacity is bounded", "[vkexec][profiling]")
{
  auto ctx = vkexec::test::require_context();
  constexpr std::uint32_t k_event_limit = 1;
  auto state = vkexec::profiling::detail::profiler_state(
    *ctx, { .max_completed_events = k_event_limit }, {}, 1.0F);
  state.publish(vkexec::profiling::profile_event{ .name = "first" });
  state.publish(vkexec::profiling::profile_event{ .name = "dropped" });
  REQUIRE(state.dropped_events() == 1);
  auto events = state.take_completed();
  REQUIRE(events.size() == k_event_limit);
  REQUIRE(events.front().name == "first");
  REQUIRE(state.take_completed().empty());
}

TEST_CASE("profiler rejects invalid query-pool options", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  REQUIRE_FALSE(vkexec::profiling::profiler::create(*ctx, { .queries_per_pool = 1 }));
  REQUIRE_FALSE(vkexec::profiling::profiler::create(*ctx, { .max_in_flight_regions = 0 }));
}
