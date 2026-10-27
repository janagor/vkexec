#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/error.hpp>
#include <vkexec/submit_scope.hpp>
#include <vkexec_profiling/detail/fence_status.hpp>
#include <vkexec_profiling/manual_capture.hpp>
#include <vkexec_profiling/profile_event.hpp>
#include <vkexec_profiling/profiler.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <limits>
#include <utility>

namespace {

auto out_of_memory_fence_status(VkDevice /*device*/, VkFence /*fence*/) -> VkResult
{ return VK_ERROR_OUT_OF_HOST_MEMORY; }
auto device_lost_fence_status(VkDevice /*device*/, VkFence /*fence*/) -> VkResult { return VK_ERROR_DEVICE_LOST; }

struct scoped_fence_status_override
{
  explicit scoped_fence_status_override(vkexec::profiling::detail::fence_status_fn replacement)
    : previous(vkexec::profiling::detail::set_fence_status_override(replacement))
  {}
  scoped_fence_status_override(scoped_fence_status_override const &) = delete;
  auto operator=(scoped_fence_status_override const &) -> scoped_fence_status_override & = delete;
  scoped_fence_status_override(scoped_fence_status_override &&) = delete;
  auto operator=(scoped_fence_status_override &&) -> scoped_fence_status_override & = delete;
  ~scoped_fence_status_override() { (void)vkexec::profiling::detail::set_fence_status_override(previous); }

  vkexec::profiling::detail::fence_status_fn previous;
};

}// namespace

TEST_CASE("manual capture reports a completed timestamp pair", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx);
  REQUIRE(profiler);
  auto const queue = ctx->compute_queue_ref();
  if (!profiler->supported(queue)) { SKIP("compute queue has no timestamp support"); }

  auto scope = vkexec::detail::submit_scope::open(*ctx, queue);
  REQUIRE(scope);
  auto capture = vkexec::profiling::manual_capture::create(*profiler, *ctx, scope->cmd, queue);
  REQUIRE(capture);
  {
    auto region = capture->scoped_region("empty-region");
    REQUIRE(region);
  }
  REQUIRE(scope->end_recording());
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence = VK_NULL_HANDLE;
  REQUIRE(vkCreateFence(ctx->device(), &fence_info, nullptr, &fence) == VK_SUCCESS);
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &scope->cmd;
  {
    auto guard = ctx->lock_queue(queue.queue);
    REQUIRE(vkQueueSubmit(queue.queue, 1, &submit_info, fence) == VK_SUCCESS);
  }
  REQUIRE(capture->mark_submitted(fence));
  REQUIRE(vkWaitForFences(ctx->device(), 1, &fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max()) == VK_SUCCESS);
  {
    auto override = scoped_fence_status_override(out_of_memory_fence_status);
    auto failed = capture->resolve_after_gpu();
    REQUIRE_FALSE(failed);
    REQUIRE(failed.error().code == vkexec::make_vk_error_code(VK_ERROR_OUT_OF_HOST_MEMORY));
  }
  REQUIRE(capture->resolve_after_gpu());
  auto events = profiler->take_completed();
  REQUIRE(events.size() == 1);
  REQUIRE(events.at(0).name == "empty-region");
  REQUIRE(events.at(0).status == vkexec::profiling::sample_status::valid);
  vkDestroyFence(ctx->device(), fence, nullptr);
}

TEST_CASE("two manual captures can be submitted before either resolves", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx, { .queries_per_pool = 4 });
  REQUIRE(profiler);
  auto const queue = ctx->compute_queue_ref();
  if (!profiler->supported(queue)) { SKIP("compute queue has no timestamp support"); }
  auto first_scope = vkexec::detail::submit_scope::open(*ctx, queue);
  auto second_scope = vkexec::detail::submit_scope::open(*ctx, queue);
  REQUIRE(first_scope);
  REQUIRE(second_scope);
  auto first = vkexec::profiling::manual_capture::create(*profiler, *ctx, first_scope->cmd, queue);
  auto second = vkexec::profiling::manual_capture::create(*profiler, *ctx, second_scope->cmd, queue);
  REQUIRE(first);
  REQUIRE(second);
  {
    auto outer = first->scoped_region("outer");
    REQUIRE(outer);
    auto inner = first->scoped_region("inner");
    REQUIRE(inner);
  }
  {
    auto region = second->scoped_region("second");
    REQUIRE(region);
  }
  REQUIRE(first_scope->end_recording());
  REQUIRE(second_scope->end_recording());
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence first_fence = VK_NULL_HANDLE;
  VkFence second_fence = VK_NULL_HANDLE;
  REQUIRE(vkCreateFence(ctx->device(), &fence_info, nullptr, &first_fence) == VK_SUCCESS);
  REQUIRE(vkCreateFence(ctx->device(), &fence_info, nullptr, &second_fence) == VK_SUCCESS);
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  {
    auto guard = ctx->lock_queue(queue.queue);
    submit_info.pCommandBuffers = &first_scope->cmd;
    REQUIRE(vkQueueSubmit(queue.queue, 1, &submit_info, first_fence) == VK_SUCCESS);
    submit_info.pCommandBuffers = &second_scope->cmd;
    REQUIRE(vkQueueSubmit(queue.queue, 1, &submit_info, second_fence) == VK_SUCCESS);
  }
  REQUIRE(first->mark_submitted(first_fence));
  REQUIRE(second->mark_submitted(second_fence));
  auto const first_completion = vkGetFenceStatus(ctx->device(), first_fence);
  if (first_completion == VK_NOT_READY) {
    REQUIRE_FALSE(first->resolve_after_gpu());
    REQUIRE_FALSE(first->discard_after_gpu());
  }
  constexpr std::uint32_t k_fence_count = 2;
  std::array<VkFence, k_fence_count> const fences{ first_fence, second_fence };
  REQUIRE(
    vkWaitForFences(ctx->device(), k_fence_count, fences.data(), VK_TRUE, std::numeric_limits<std::uint64_t>::max())
    == VK_SUCCESS);
  REQUIRE(second->resolve_after_gpu());
  REQUIRE(first->resolve_after_gpu());
  auto events = profiler->take_completed();
  REQUIRE(events.size() == 3);
  REQUIRE(events.at(0).submission_id != events.at(1).submission_id);
  REQUIRE(events.at(0).submission_id > events.at(1).submission_id);
  REQUIRE(events.at(1).submission_id == events.at(2).submission_id);
  auto third_scope = vkexec::detail::submit_scope::open(*ctx, queue);
  REQUIRE(third_scope);
  auto third = vkexec::profiling::manual_capture::create(*profiler, *ctx, third_scope->cmd, queue);
  REQUIRE(third);
  {
    auto region = third->scoped_region("reused");
    REQUIRE(region);
  }
  REQUIRE(third_scope->end_recording());
  VkFence third_fence = VK_NULL_HANDLE;
  REQUIRE(vkCreateFence(ctx->device(), &fence_info, nullptr, &third_fence) == VK_SUCCESS);
  submit_info.pCommandBuffers = &third_scope->cmd;
  {
    auto guard = ctx->lock_queue(queue.queue);
    REQUIRE(vkQueueSubmit(queue.queue, 1, &submit_info, third_fence) == VK_SUCCESS);
  }
  REQUIRE(third->mark_submitted(third_fence));
  REQUIRE(
    vkWaitForFences(ctx->device(), 1, &third_fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max()) == VK_SUCCESS);
  REQUIRE(third->discard_after_gpu());
  REQUIRE(profiler->take_completed().empty());
  vkDestroyFence(ctx->device(), first_fence, nullptr);
  vkDestroyFence(ctx->device(), second_fence, nullptr);
  vkDestroyFence(ctx->device(), third_fence, nullptr);
}

TEST_CASE("device loss makes a submitted capture terminal", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx);
  REQUIRE(profiler);
  auto const queue = ctx->compute_queue_ref();
  if (!profiler->supported(queue)) { SKIP("compute queue has no timestamp support"); }
  auto scope = vkexec::detail::submit_scope::open(*ctx, queue);
  REQUIRE(scope);
  auto capture = vkexec::profiling::manual_capture::create(*profiler, *ctx, scope->cmd, queue);
  REQUIRE(capture);
  {
    auto region = capture->scoped_region("lost");
    REQUIRE(region);
  }
  REQUIRE(scope->end_recording());
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence = VK_NULL_HANDLE;
  REQUIRE(vkCreateFence(ctx->device(), &fence_info, nullptr, &fence) == VK_SUCCESS);
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &scope->cmd;
  {
    auto guard = ctx->lock_queue(queue.queue);
    REQUIRE(vkQueueSubmit(queue.queue, 1, &submit_info, fence) == VK_SUCCESS);
  }
  REQUIRE(capture->mark_submitted(fence));
  REQUIRE(vkWaitForFences(ctx->device(), 1, &fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max()) == VK_SUCCESS);
  {
    auto override = scoped_fence_status_override(device_lost_fence_status);
    auto failed = capture->resolve_after_gpu();
    REQUIRE_FALSE(failed);
    REQUIRE(failed.error().code == vkexec::make_vk_error_code(VK_ERROR_DEVICE_LOST));
  }
  REQUIRE(profiler->take_completed().empty());
  REQUIRE_FALSE(capture->resolve_after_gpu());
  vkDestroyFence(ctx->device(), fence, nullptr);
}

TEST_CASE("unsubmitted manual captures can be moved and abandoned", "[vkexec][profiling][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto profiler = vkexec::profiling::profiler::create(*ctx);
  REQUIRE(profiler);// NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
  auto const queue = ctx->compute_queue_ref();
  if (!profiler->supported(queue)) { SKIP("compute queue has no timestamp support"); }
  auto scope = vkexec::detail::submit_scope::open(*ctx, queue);
  REQUIRE(scope);
  {
    auto capture = vkexec::profiling::manual_capture::create(*profiler, *ctx, scope->cmd, queue);
    REQUIRE(capture);
    {
      auto region = capture->scoped_region("abandoned");
      REQUIRE(region);
    }
    auto moved = std::move(*capture);
    REQUIRE_FALSE(moved.resolve_after_gpu());
  }
  REQUIRE(profiler->take_completed().empty());
  // The recorded buffer is abandoned with its submit scope.
}
