#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/object_synchronization.hpp>
#include <vkexec/result.hpp>
#include <vkexec/submit_scope.hpp>
#include <vulkan/vulkan_core.h>

#include <array>
#include <atomic>
#include <barrier>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

namespace {

constexpr int k_thread_count = 4;
constexpr int k_default_iterations = 8;

[[nodiscard]] auto stress_iterations() -> int
{
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  if (auto const *value = std::getenv("VKEXEC_CONCURRENCY_ITERATIONS")) {
    std::string_view const input{ value };
    int parsed = 0;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    auto const result = std::from_chars(input.data(), input.data() + input.size(), parsed);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    if (result.ec == std::errc{} && result.ptr == input.data() + input.size() && parsed > 0) { return parsed; }
  }
  return k_default_iterations;
}

}// namespace

TEST_CASE("concurrent submit scopes reuse command pools", "[vkexec][concurrency][gpu]")
{
  auto ctx = vkexec::test::require_context();
  std::barrier start{ k_thread_count };
  std::atomic<int> completed{ 0 };
  auto record = [&]() -> void {
    start.arrive_and_wait();
    for (int iteration = 0; iteration < stress_iterations(); ++iteration) {
      auto opened = vkexec::detail::submit_scope::open(*ctx);
      if (!opened) { return; }
      auto scope = vkexec::expected_take(opened);
      if (!scope.end_recording()) { return; }
      auto outcome = vkexec::test::sync_wait_sender(vkexec::detail::submit_and_wait(std::move(scope)));
      if (!vkexec::test::sync_wait_completed(outcome)) { return; }
      completed.fetch_add(1, std::memory_order_relaxed);
    }
  };
  std::array<std::thread, k_thread_count> threads;
  for (auto &thread : threads) { thread = std::thread(record); }
  for (auto &thread : threads) { thread.join(); }
  REQUIRE(completed.load() == k_thread_count * stress_iterations());
}

TEST_CASE("queue guards and host enqueue withstand contention", "[vkexec][concurrency][gpu]")
{
  auto ctx = vkexec::test::require_context();
  std::barrier start{ k_thread_count };
  std::atomic<int> enqueued{ 0 };
  std::atomic<int> executed{ 0 };
  auto work = [&]() -> void {
    start.arrive_and_wait();
    for (int iteration = 0; iteration < stress_iterations(); ++iteration) {
      {
        auto guard = ctx->lock_queue(ctx->compute_queue());
      }
      auto result =
        ctx->enqueue_host([&executed]() noexcept -> void { executed.fetch_add(1, std::memory_order_relaxed); });
      if (result) { enqueued.fetch_add(1, std::memory_order_relaxed); }
    }
  };
  std::array<std::thread, k_thread_count> threads;
  for (auto &thread : threads) { thread = std::thread(work); }
  for (auto &thread : threads) { thread.join(); }
  REQUIRE(enqueued.load() == k_thread_count * stress_iterations());
  // The context destructor drains accepted host work.
  ctx.reset();
  REQUIRE(executed.load() == enqueued.load());
}

TEST_CASE("direct command buffers submit concurrently", "[vkexec][concurrency][gpu]")
{
  auto ctx = vkexec::test::require_context();
  std::barrier start{ k_thread_count };
  std::atomic<int> completed{ 0 };
  auto submit = [&]() -> void {
    start.arrive_and_wait();
    for (int iteration = 0; iteration < stress_iterations(); ++iteration) {
      auto allocated = ctx->allocate_command_buffer();
      if (!allocated) { return; }
      VkCommandBuffer cmd = vkexec::expected_take(allocated);
      VkCommandBufferBeginInfo begin{};
      begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      if (vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS && vkEndCommandBuffer(cmd) == VK_SUCCESS
          && ctx->submit_and_wait(cmd).has_value()) {
        completed.fetch_add(1, std::memory_order_relaxed);
      }
      ctx->free_command_buffer(cmd);
    }
  };
  std::array<std::thread, k_thread_count> threads;
  for (auto &thread : threads) { thread = std::thread(submit); }
  for (auto &thread : threads) { thread.join(); }
  REQUIRE(completed.load() == k_thread_count * stress_iterations());
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("descriptor pools allocate under contention", "[vkexec][concurrency][gpu]")
{
  bool same_pool = false;
  SECTION("same descriptor pool") { same_pool = true; }
  SECTION("distinct descriptor pools") {}

  auto ctx = vkexec::test::require_context();
  auto const iterations = stress_iterations();
  auto const pool_count = same_pool ? 1 : k_thread_count;
  auto const allocations_per_pool = iterations * (same_pool ? k_thread_count : 1);
  VkDescriptorSetLayoutBinding binding{};
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = 1;
  layout_info.pBindings = &binding;
  VkDescriptorSetLayout layout{ VK_NULL_HANDLE };
  REQUIRE(vkCreateDescriptorSetLayout(ctx->device(), &layout_info, nullptr, &layout) == VK_SUCCESS);

  VkDescriptorPoolSize const size{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = static_cast<std::uint32_t>(allocations_per_pool) };
  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = static_cast<std::uint32_t>(allocations_per_pool);
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &size;
  std::array<VkDescriptorPool, k_thread_count> pools{};
  for (int index = 0; index < pool_count; ++index) {
    auto &pool = pools.at(static_cast<std::size_t>(index));
    REQUIRE(vkCreateDescriptorPool(ctx->device(), &pool_info, nullptr, &pool) == VK_SUCCESS);
    REQUIRE(vkexec::detail::descriptor_pool_access::register_owned(*ctx, pool).has_value());
  }

  std::barrier start{ k_thread_count };
  std::atomic<int> allocated{ 0 };
  auto allocate = [&](VkDescriptorPool pool) -> void {
    start.arrive_and_wait();
    for (int iteration = 0; iteration < iterations; ++iteration) {
      VkDescriptorSetAllocateInfo info{};
      info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      info.descriptorPool = pool;
      info.descriptorSetCount = 1;
      info.pSetLayouts = &layout;
      VkDescriptorSet set{ VK_NULL_HANDLE };
      auto const guard = vkexec::detail::descriptor_pool_access::lock(*ctx, pool);
      if (vkAllocateDescriptorSets(ctx->device(), &info, &set) == VK_SUCCESS) {
        allocated.fetch_add(1, std::memory_order_relaxed);
      }
    }
  };
  std::array<std::thread, k_thread_count> threads;
  for (int index = 0; index < k_thread_count; ++index) {
    threads.at(static_cast<std::size_t>(index)) =
      std::thread(allocate, pools.at(static_cast<std::size_t>(same_pool ? 0 : index)));
  }
  for (auto &thread : threads) { thread.join(); }

  for (int index = 0; index < pool_count; ++index) {
    auto const &pool = pools.at(static_cast<std::size_t>(index));
    auto const guard = vkexec::detail::descriptor_pool_access::lock(*ctx, pool);
    vkDestroyDescriptorPool(ctx->device(), pool, nullptr);
  }
  vkDestroyDescriptorSetLayout(ctx->device(), layout, nullptr);
  REQUIRE(allocated.load() == k_thread_count * iterations);
}
