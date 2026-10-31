#include "recording_buffer.hpp"
#include "recording_compute.hpp"
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_message.hpp>
#include <cstring>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <stdexec/execution.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/pass_execution.hpp>
#include <vkexec/detail/pass_types.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/recorded_graph.hpp>
#include <vkexec/result.hpp>
#include <vkexec/submit_scope.hpp>
#include <vulkan/vulkan_core.h>
#include <vkexec/barrier.hpp>
#include <vkexec/barrier_params.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/bind_resources.hpp>
#include <vkexec/record_into.hpp>

namespace {
constexpr VkDeviceSize k_hazard_size = 16;
constexpr VkDeviceSize k_overlapping_offset = 8;
constexpr VkDeviceSize k_disjoint_offset = 32;
constexpr VkDeviceSize k_compute_word_offset = 12;
constexpr VkDeviceSize k_compute_word_size = sizeof(std::uint32_t);
constexpr std::uint32_t k_image_extent = 16;
constexpr std::size_t k_readback_word_count = vkexec::test::recording_buffer::k_byte_size / sizeof(std::uint32_t);

struct observed_step
{
  int *recorded;
  int *completed;
  int *committed;
  std::uint64_t *identity;
  bool fail_recording{};
  auto
    record(vkexec::context &ctx, VkCommandBuffer command_buffer, vkexec::detail::pass_cleanup &cleanup,
      vkexec::pass_recording_info const &info) const -> vkexec::status
  {
    (void)ctx;
    (void)command_buffer;
    (void)cleanup;
    ++*recorded;
    auto const submission_id = info.ensure_submission_id();
    if (*identity == 0) { *identity = submission_id; }
    CHECK(*identity == submission_id);
    if (fail_recording) { return vkexec::fail(vkexec::errc::invalid_argument, "test recording failure"); }
    return {};
  }
  auto after_gpu() const -> void { ++*completed; }
  auto commit_after_gpu() const -> void { ++*committed; }
};
}// namespace

TEST_CASE("external recording retains callbacks until explicit retirement", "[vkexec][record_into]")
{
  auto ctx = vkexec::test::require_context();
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(opened);
  auto scope = vkexec::expected_take(opened);
  int recorded{};
  int completed{};
  int committed{};
  std::uint64_t identity{};
  auto step = observed_step{ .recorded = &recorded, .completed = &completed, .committed = &committed, .identity = &identity };
  auto graph =
    vkexec::pass_graph_sender<observed_step, observed_step>{ .state = vkexec::detail::context_access::state(*ctx),
      .steps = { step, step },
      .step_queues = { {}, {} },
      .step_predecessors = { {}, { 0 } },
      .presentation = {},
      .current_queue = {} };
  auto outcome = vkexec::record_into(*ctx, scope.cmd, graph, { scope.queue });
  REQUIRE(outcome);
  CHECK(recorded == 2);
  CHECK(completed == 0);
  CHECK(committed == 0);
  vkexec::recorded_graph moved = vkexec::expected_take(outcome);
  CHECK(!(moved.complete()));
  CHECK(completed == 0);
  REQUIRE(scope.end_recording());
  REQUIRE(ctx->submit_and_wait(scope.cmd, scope.queue));
  REQUIRE(moved.mark_submitted());
  CHECK(!(moved.mark_submitted()));
  CHECK(!(moved.abandon()));
  REQUIRE(moved.complete());
  CHECK(completed == 2);
  CHECK(committed == 2);
  CHECK(!(moved.complete()));
  CHECK(!(moved.abandon()));
}

TEST_CASE("external static expressions and move only dynamic graphs", "[vkexec][record_into]")
{
  auto ctx = vkexec::test::require_context();
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(opened);
  auto scope = vkexec::expected_take(opened);
  int calls{};
  auto expression = stdexec::schedule(ctx->get_scheduler())
                    | vkexec::custom_pass(vkexec::uses(), [&calls](VkCommandBuffer command_buffer) -> void { (void)command_buffer; ++calls; });
  auto first = vkexec::record_into(*ctx, scope.cmd, std::move(expression), { scope.queue });
  REQUIRE(first);
  auto dynamic = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  dynamic.append(vkexec::custom_pass(
    vkexec::uses(), [owned = std::make_unique<int>(1), &calls](VkCommandBuffer command_buffer) -> void { (void)command_buffer; calls += *owned; }));
  auto second = vkexec::record_into(*ctx, scope.cmd, std::move(dynamic), { scope.queue });
  REQUIRE(second);
  CHECK(calls == 2);
  REQUIRE(scope.end_recording());
  REQUIRE(vkResetCommandBuffer(scope.cmd, 0) == VK_SUCCESS);
  REQUIRE(first->abandon());
  REQUIRE(second->abandon());
  CHECK(!(first->abandon()));
}

TEST_CASE("external recording validates inputs and supports empty graphs", "[vkexec][record_into]")
{
  auto ctx = vkexec::test::require_context();
  auto graph = vkexec::pass_graph_sender<>{};
  graph.state = vkexec::detail::context_access::state(*ctx);
  CHECK(!(vkexec::record_into(*ctx, VK_NULL_HANDLE, graph, { ctx->compute_queue_ref() })));
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(opened);
  auto scope = vkexec::expected_take(opened);
  auto foreign = graph;
  foreign.state.reset();
  CHECK(!(vkexec::record_into(*ctx, scope.cmd, foreign, { scope.queue })));
  auto wrong_queue = scope.queue;
  ++wrong_queue.family;
  CHECK(!(vkexec::record_into(*ctx, scope.cmd, graph, { wrong_queue })));
  graph.presentation.emplace();
  CHECK(!(vkexec::record_into(*ctx, scope.cmd, graph, { scope.queue })));
  graph.presentation.reset();
  auto empty = vkexec::record_into(*ctx, scope.cmd, graph, { scope.queue });
  REQUIRE(empty);
  REQUIRE(scope.end_recording());
  REQUIRE(vkResetCommandBuffer(scope.cmd, 0) == VK_SUCCESS);
  REQUIRE(empty->abandon());
}

TEST_CASE("external recording stops on failure without invoking completion", "[vkexec][record_into]")
{
  auto ctx = vkexec::test::require_context();
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(opened);
  auto scope = vkexec::expected_take(opened);
  int recorded{};
  int completed{};
  int committed{};
  std::uint64_t identity{};
  auto step = observed_step{ .recorded = &recorded, .completed = &completed, .committed = &committed, .identity = &identity };
  auto failing = step;
  failing.fail_recording = true;
  auto graph = vkexec::pass_graph_sender<observed_step, observed_step, observed_step>{
    .state = vkexec::detail::context_access::state(*ctx),
    .steps = { step, failing, step },
    .step_queues = {},
    .step_predecessors = {},
    .presentation = {},
    .current_queue = {}
  };
  auto failed_recording = vkexec::record_into(*ctx, scope.cmd, std::move(graph), { scope.queue });
  CHECK(!(failed_recording));
  CHECK(recorded == 2);
  CHECK(completed == 0);
  CHECK(committed == 0);
  REQUIRE(vkResetCommandBuffer(scope.cmd, 0) == VK_SUCCESS);
}


TEST_CASE("external recording synchronizes transfer hazards between surrounding commands", "[vkexec][record_into]")
{
  auto ctx = vkexec::test::require_context();
  auto allocated = vkexec::test::recording_buffer::make(*ctx);
  REQUIRE(allocated);
  auto buffer = std::move(*allocated);
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(opened);
  auto scope = vkexec::expected_take(opened);
  vkexec::buffer_barrier_params const boundary{ .buffer = buffer->buffer,
    .size = buffer->k_byte_size,
    .src_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    .dst_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
    .dst_access = VK_ACCESS_2_TRANSFER_WRITE_BIT };
  vkCmdFillBuffer(scope.cmd, buffer->buffer, 0, buffer->k_byte_size, 0);
  REQUIRE(vkexec::buffer_barrier(*ctx, scope.cmd, boundary));
  auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
  for (std::uint32_t value = 1; value <= 3; ++value) {
    VkDeviceSize const offset = value < 3 ? (value - 1) * 32 : 0;
    VkDeviceSize const size = value < 3 ? 32 : 64;
    auto const use = vkexec::write(buffer->buffer, offset, size, vkexec::buffer_usage::transfer_destination);
    (void)graph.append_after(
      {}, vkexec::custom_pass(vkexec::uses(use), [handle = buffer->buffer, value, offset, size](VkCommandBuffer cmd) -> void {
        vkCmdFillBuffer(cmd, handle, offset, size, value);
      }));
  }
  auto recorded = vkexec::record_into(*ctx, scope.cmd, std::move(graph), { scope.queue });
  if (!recorded) {
    INFO(recorded.error().message());
    REQUIRE(recorded);
  }
  REQUIRE(recorded);
  REQUIRE(vkexec::buffer_barrier(*ctx, scope.cmd, boundary));
  vkCmdFillBuffer(scope.cmd, buffer->buffer, 0, buffer->k_byte_size, 4);
  REQUIRE(scope.end_recording());
  REQUIRE(ctx->submit_and_wait(scope.cmd, scope.queue));
  REQUIRE(recorded->mark_submitted());
  REQUIRE(recorded->complete());
}

TEST_CASE("external recording rejects queue affinity and invalid topology before commands", "[vkexec][record_into]")
{
  auto ctx = vkexec::test::require_context();
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(opened);
  auto scope = vkexec::expected_take(opened);
  int calls{};
  auto make_graph = [&]() -> auto {
    auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
    graph.append(vkexec::custom_pass(vkexec::uses(), [&calls](VkCommandBuffer command_buffer) -> void { (void)command_buffer; ++calls; }));
    return graph;
  };
  auto graph = make_graph();
  auto wrong = scope.queue;
  ++wrong.family;
  graph.step_queues = { wrong };
  auto wrong_queue_recording = vkexec::record_into(*ctx, scope.cmd, std::move(graph), { scope.queue });
  CHECK(!(wrong_queue_recording));
  graph = make_graph();
  graph.step_predecessors = { { 0 } };
  auto invalid_topology_recording = vkexec::record_into(*ctx, scope.cmd, std::move(graph), { scope.queue });
  CHECK(!(invalid_topology_recording));
  CHECK(calls == 0);
  REQUIRE(vkResetCommandBuffer(scope.cmd, 0) == VK_SUCCESS);
}

TEST_CASE("shared preparation generates declared buffer hazard barriers", "[vkexec][record_into][sync]")
{
  auto ctx = vkexec::test::require_context();
  auto allocated = vkexec::test::recording_buffer::make(*ctx);
  REQUIRE(allocated);
  auto *const handle = (*allocated)->buffer;
  struct hazard
  {
    vkexec::buffer_usage source_usage;
    vkexec::buffer_usage destination_usage;
    vkexec::resource_access source;
    vkexec::resource_access destination;
  };
  std::array const hazards{ hazard{ .source_usage = vkexec::buffer_usage::storage_compute,
      .destination_usage = vkexec::buffer_usage::storage_compute,
      .source = vkexec::resource_access::write,
      .destination = vkexec::resource_access::read },
    hazard{ .source_usage = vkexec::buffer_usage::storage_compute,
      .destination_usage = vkexec::buffer_usage::storage_compute,
      .source = vkexec::resource_access::read,
      .destination = vkexec::resource_access::write },
    hazard{ .source_usage = vkexec::buffer_usage::storage_compute,
      .destination_usage = vkexec::buffer_usage::storage_compute,
      .source = vkexec::resource_access::write,
      .destination = vkexec::resource_access::write },
    hazard{ .source_usage = vkexec::buffer_usage::transfer_destination,
      .destination_usage = vkexec::buffer_usage::storage_compute,
      .source = vkexec::resource_access::write,
      .destination = vkexec::resource_access::read },
    hazard{ .source_usage = vkexec::buffer_usage::storage_compute,
      .destination_usage = vkexec::buffer_usage::indirect,
      .source = vkexec::resource_access::write,
      .destination = vkexec::resource_access::read },
    hazard{ .source_usage = vkexec::buffer_usage::storage_compute,
      .destination_usage = vkexec::buffer_usage::transfer_source,
      .source = vkexec::resource_access::write,
      .destination = vkexec::resource_access::read } };
  for (auto const &hazard : hazards) {
    auto make_step = [&](vkexec::buffer_usage usage, vkexec::resource_access access, VkDeviceSize offset) -> auto {
      return vkexec::custom_pass(vkexec::uses(vkexec::buffer_use{
                                   .buffer = handle, .offset = offset, .size = k_hazard_size, .usage = usage, .access = access }),
        [](VkCommandBuffer command_buffer) -> void { (void)command_buffer;});
    };
    for (bool const overlap : { false, true }) {
      auto graph = vkexec::make_dynamic_pass_graph(stdexec::schedule(ctx->get_scheduler()));
      (void)graph.append_after({}, make_step(hazard.source_usage, hazard.source, 0));
      (void)graph.append_after({}, make_step(hazard.destination_usage, hazard.destination, overlap ? k_overlapping_offset : k_disjoint_offset));
      auto prepared = vkexec::detail::prepare_graph(
        *ctx, graph.steps, graph.step_queues, graph.step_predecessors, ctx->compute_queue_ref());
      REQUIRE(prepared);
      CHECK(prepared->synchronization.steps.at(1).buffers_before.empty() == !overlap);
      CHECK(prepared->execution.batches.at(1).predecessors.empty() == !overlap);
    }
  }
}


TEST_CASE("external compute and indirect passes retain temporary descriptor bindings", "[vkexec][record_into][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocated = vkexec::test::recording_buffer::make(*ctx);
  REQUIRE(allocated);
  auto buffer = std::move(*allocated);
  auto readback_allocation = vkexec::test::recording_buffer::make(*ctx, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  REQUIRE(readback_allocation);
  auto readback = std::move(*readback_allocation);
  auto pipeline = vkexec::test::recording_compute::make(*ctx, buffer->buffer, buffer->k_byte_size);
  REQUIRE(pipeline);
  auto compute = std::move(*pipeline);
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(opened);
  auto scope = vkexec::expected_take(opened);
  auto const table = vkexec::bindings(
    vkexec::resource_binding{ .slot = 0, .resource = vkexec::buffer_resource(buffer->buffer, buffer->k_byte_size) });
  // Binding allocates a temporary descriptor set. Subsequent set-less dispatches
  // reuse it; it must remain valid after record_into returns and until retirement.
  auto bind =
    vkexec::bind_compute(compute->pipeline)
      .with_resources({}, { vkexec::read_write(buffer->buffer, k_compute_word_offset, k_compute_word_size, vkexec::buffer_usage::storage_compute) });
  auto graph =
    stdexec::schedule(ctx->get_scheduler())
    | vkexec::custom_pass(
      vkexec::uses(vkexec::write(buffer->buffer, 0, buffer->k_byte_size, vkexec::buffer_usage::transfer_destination)),
      [handle = buffer->buffer](VkCommandBuffer cmd) -> void { vkCmdFillBuffer(cmd, handle, 0, vkexec::test::recording_buffer::k_byte_size, 1); })
    | vkexec::bind_resources(compute->pipeline, table) | vkexec::compute_pass(bind, vkexec::dispatch{})
    | vkexec::compute_pass(bind, vkexec::dispatch{})
    | vkexec::compute_pass(bind, vkexec::indirect_dispatch{ .buffer = buffer->buffer, .offset = 0 })
    | vkexec::custom_pass(
      vkexec::uses(vkexec::read(buffer->buffer, 0, buffer->k_byte_size, vkexec::buffer_usage::transfer_source),
        vkexec::write(readback->buffer, 0, readback->k_byte_size, vkexec::buffer_usage::transfer_destination)),
      [source = buffer->buffer, destination = readback->buffer](VkCommandBuffer cmd) -> void {
        VkBufferCopy const region{ .srcOffset = 0, .dstOffset = 0, .size = vkexec::test::recording_buffer::k_byte_size };
        vkCmdCopyBuffer(cmd, source, destination, 1, &region);
      });
  auto recorded = vkexec::record_into(*ctx, scope.cmd, std::move(graph), { scope.queue });
  if (!recorded) {
    INFO(recorded.error().message());
    REQUIRE(recorded);
  }
  REQUIRE(recorded);
  vkexec::recorded_graph destination;
  destination = vkexec::expected_take(recorded);
  // The graph's final transfer write becomes visible to the caller's host read.
  REQUIRE(vkexec::buffer_barrier(*ctx,
    scope.cmd,
    vkexec::buffer_barrier_params{ .buffer = readback->buffer,
      .size = readback->k_byte_size,
      .src_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .dst_stage = VK_PIPELINE_STAGE_2_HOST_BIT,
      .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dst_access = VK_ACCESS_2_HOST_READ_BIT }));
  REQUIRE(scope.end_recording());
  REQUIRE(ctx->submit_and_wait(scope.cmd, scope.queue));
  REQUIRE(destination.mark_submitted());
  void *mapped{};
  REQUIRE(vkMapMemory(ctx->device(), readback->memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS);
  VkMappedMemoryRange range{};
  range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  range.memory = readback->memory;
  range.size = VK_WHOLE_SIZE;
  auto const invalidated = vkInvalidateMappedMemoryRanges(ctx->device(), 1, &range);
  std::array<std::uint32_t, k_readback_word_count> values{};
  if (invalidated == VK_SUCCESS) { std::memcpy(values.data(), mapped, sizeof(values)); }
  vkUnmapMemory(ctx->device(), readback->memory);
  CHECK(invalidated == VK_SUCCESS);
  for (std::size_t index{}; index < values.size(); ++index) { CHECK(values.at(index) == (index == 3 ? 4U : 1U)); }
  REQUIRE(destination.complete());
}


namespace {
struct recording_image
{
  VkDevice device{};
  VkImage image{};
  VkDeviceMemory memory{};
  explicit recording_image(VkDevice value) : device(value) {}
  recording_image(recording_image const &) = delete;
  auto operator=(recording_image const &) -> recording_image & = delete;
  recording_image(recording_image &&) = delete;
  auto operator=(recording_image &&) -> recording_image & = delete;
  ~recording_image()
  {
    if (image != VK_NULL_HANDLE) { vkDestroyImage(device, image, nullptr); }
    if (memory != VK_NULL_HANDLE) { vkFreeMemory(device, memory, nullptr); }
  }
};
}// namespace

TEST_CASE("external image transitions cover mip and layer subresources", "[vkexec][record_into][gpu]")
{
  auto ctx = vkexec::test::require_context();
  recording_image image{ ctx->device() };
  VkImageCreateInfo info{ .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    .pNext = nullptr,
    .flags = {},
    .imageType = VK_IMAGE_TYPE_2D,
    .format = VK_FORMAT_UNDEFINED,
    .extent = {},
    .mipLevels = {},
    .arrayLayers = {},
    .samples = VK_SAMPLE_COUNT_1_BIT,
    .tiling = VK_IMAGE_TILING_OPTIMAL,
    .usage = {},
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    .queueFamilyIndexCount = {},
    .pQueueFamilyIndices = nullptr,
    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = VK_FORMAT_R8G8B8A8_UNORM;
  info.extent = { .width = k_image_extent, .height = k_image_extent, .depth = 1 };
  info.mipLevels = 2;
  info.arrayLayers = 2;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = VkImageUsageFlags{ VK_IMAGE_USAGE_TRANSFER_DST_BIT }
               | VkImageUsageFlags{ VK_IMAGE_USAGE_SAMPLED_BIT } | VkImageUsageFlags{ VK_IMAGE_USAGE_STORAGE_BIT };
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  REQUIRE(vkCreateImage(ctx->device(), &info, nullptr, &image.image) == VK_SUCCESS);
  VkMemoryRequirements requirements{};
  vkGetImageMemoryRequirements(ctx->device(), image.image, &requirements);
  VkPhysicalDeviceMemoryProperties properties{};
  vkGetPhysicalDeviceMemoryProperties(ctx->physical_device(), &properties);
  VkMemoryAllocateInfo allocation{};
  allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocation.allocationSize = requirements.size;
  for (std::uint32_t index{}; index < properties.memoryTypeCount; ++index) {
    if ((requirements.memoryTypeBits & (1U << index)) != 0) {
      allocation.memoryTypeIndex = index;
      break;
    }
  }
  REQUIRE(vkAllocateMemory(ctx->device(), &allocation, nullptr, &image.memory) == VK_SUCCESS);
  REQUIRE(vkBindImageMemory(ctx->device(), image.image, image.memory, 0) == VK_SUCCESS);
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(opened);
  auto scope = vkexec::expected_take(opened);
  VkImageSubresourceRange const all{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 2, .baseArrayLayer = 0, .layerCount = 2 };
  VkImageSubresourceRange const part{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 1, .levelCount = 1, .baseArrayLayer = 1, .layerCount = 1 };
  auto graph =
    stdexec::schedule(ctx->get_scheduler())
    | vkexec::custom_pass(vkexec::uses(vkexec::write(image.image, all, vkexec::image_usage::transfer_destination)),
      [handle = image.image, all](VkCommandBuffer cmd) -> void {
        VkClearColorValue const color{};
        vkCmdClearColorImage(cmd, handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &all);
      })
    | vkexec::custom_pass(
      vkexec::uses(vkexec::read(image.image, part, vkexec::image_usage::sampled_compute)), [](VkCommandBuffer command_buffer) -> void { (void)command_buffer;})
    | vkexec::custom_pass(
      vkexec::uses(vkexec::write(image.image, part, vkexec::image_usage::storage_compute)), [](VkCommandBuffer command_buffer) -> void { (void)command_buffer;});
  auto recorded = vkexec::record_into(*ctx, scope.cmd, std::move(graph), { scope.queue });
  REQUIRE(recorded);
  REQUIRE(scope.end_recording());
  REQUIRE(ctx->submit_and_wait(scope.cmd, scope.queue));
  REQUIRE(recorded->mark_submitted());
  REQUIRE(recorded->complete());
}

TEST_CASE("abandonment releases steps without completion callbacks", "[vkexec][record_into]")
{
  auto ctx = vkexec::test::require_context();
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(opened);
  auto scope = vkexec::expected_take(opened);
  int recorded{};
  int completed{};
  int committed{};
  std::uint64_t identity{};
  auto graph = stdexec::schedule(ctx->get_scheduler())
               | vkexec::make_pass_adaptor(observed_step{ .recorded = &recorded, .completed = &completed, .committed = &committed, .identity = &identity });
  auto outcome = vkexec::record_into(*ctx, scope.cmd, std::move(graph), { scope.queue });
  REQUIRE(outcome);
  CHECK(recorded == 1);
  REQUIRE(vkResetCommandBuffer(scope.cmd, 0) == VK_SUCCESS);
  REQUIRE(outcome->abandon());
  CHECK(completed == 0);
  CHECK(committed == 0);
  CHECK(!(outcome->complete()));
}

namespace {
struct temporary_descriptor_step
{
  vkexec::handles::compute_pipeline const *pipeline;
  VkDescriptorSet *set;
  auto record(vkexec::context &ctx, VkCommandBuffer command_buffer, vkexec::detail::pass_cleanup &cleanup) const -> vkexec::status
  {
    (void)command_buffer;
    auto allocated = vkexec::allocate_compute_set(ctx, *pipeline);
    if (!allocated) { return vkexec::fail(allocated); }
    *set = *allocated;
    cleanup.track(pipeline->descriptor_pool, *set);
    return {};
  }
};
}// namespace

TEST_CASE("temporary descriptor cleanup survives return and recording failures do not exhaust pools",
  "[vkexec][record_into][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocated = vkexec::test::recording_buffer::make(*ctx);
  REQUIRE(allocated);
  auto buffer = std::move(*allocated);
  auto pipeline = vkexec::test::recording_compute::make(*ctx, buffer->buffer, buffer->k_byte_size);
  REQUIRE(pipeline);
  auto compute = std::move(*pipeline);
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  REQUIRE(opened);
  auto scope = vkexec::expected_take(opened);
  VkDescriptorSet set{};
  auto make_descriptor = [&]() -> auto {
    return vkexec::make_pass_adaptor(temporary_descriptor_step{ .pipeline = &compute->pipeline, .set = &set });
  };
  auto graph = stdexec::schedule(ctx->get_scheduler()) | make_descriptor();
  auto recorded = vkexec::record_into(*ctx, scope.cmd, std::move(graph), { scope.queue });
  REQUIRE(recorded);
  std::array const bindings{ vkexec::storage_binding{
    .buffer = buffer->buffer, .byte_size = buffer->k_byte_size, .binding = 0 } };
  // Updating this set after return is legal only while the cleanup allocation is live.
  vkexec::write_storage_descriptors(ctx->device(), set, bindings);
  REQUIRE(scope.end_recording());
  REQUIRE(ctx->submit_and_wait(scope.cmd, scope.queue));
  vkexec::write_storage_descriptors(ctx->device(), set, bindings);
  REQUIRE(recorded->mark_submitted());
  REQUIRE(recorded->complete());
  // The native fixture pool holds 64 sets. More failures than its capacity must
  // still reach the intended failing step rather than exhausting allocations.
  int calls{};
  int completed{};
  int committed{};
  constexpr int k_failure_attempt_count = 80;
  for (int index{}; index < k_failure_attempt_count; ++index) {
    REQUIRE(vkResetCommandBuffer(scope.cmd, 0) == VK_SUCCESS);
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQUIRE(vkBeginCommandBuffer(scope.cmd, &begin) == VK_SUCCESS);
    std::uint64_t identity{};
    auto attempt = stdexec::schedule(ctx->get_scheduler()) | make_descriptor()
                   | vkexec::make_pass_adaptor(observed_step{ .recorded = &calls, .completed = &completed, .committed = &committed, .identity = &identity, .fail_recording = true });
    auto failed = vkexec::record_into(*ctx, scope.cmd, std::move(attempt), { scope.queue });
    REQUIRE(!(failed));
    CHECK(failed.error().detail == "test recording failure");
  }
  REQUIRE(vkResetCommandBuffer(scope.cmd, 0) == VK_SUCCESS);
  CHECK(calls == k_failure_attempt_count);
  CHECK(completed == 0);
  CHECK(committed == 0);
}
