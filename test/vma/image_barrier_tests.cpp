#include "test_helpers.hpp"
#include "vma_test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/context.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/result.hpp>
#include <vkexec_vma/image.hpp>

#include <stdexec/execution.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>

namespace {

constexpr std::uint32_t k_width = 32;
constexpr std::uint32_t k_height = 32;
constexpr std::uint32_t k_pass_extent = 16;
constexpr std::uint32_t k_array_layers = 6;
constexpr VkDeviceSize k_buffer_size = 256;
constexpr VkDeviceSize k_barrier_offset = 64;
constexpr VkDeviceSize k_barrier_size = 128;

}// namespace

TEST_CASE("image_barrier transitions a color image to general", "[vkexec][image][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto img = vkexec::test::sync_wait_value(vkexec::vma::factory::make_image(allocator,
    vkexec::vma::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::vma::image_usage::color_storage,
    }));

  auto cmd_result = ctx->allocate_command_buffer();
  REQUIRE(cmd_result.has_value());
  auto *cmd = vkexec::expected_take(cmd_result);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);

  REQUIRE(vkexec::image_barrier(*ctx,
    cmd,
    {
      .image = img.handle(),
      .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
      .new_layout = VK_IMAGE_LAYOUT_GENERAL,
      .src_stage = VK_PIPELINE_STAGE_2_NONE,
      .dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .src_access = 0,
      .dst_access = VK_ACCESS_2_SHADER_WRITE_BIT,
    }));

  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
  REQUIRE(ctx->submit_and_wait(cmd));
  ctx->free_command_buffer(cmd);
}

TEST_CASE("resource barriers record finite buffer and image ranges", "[vkexec][barrier][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto buffer = vkexec::test::sync_wait_value(vkexec::allocate_buffer(
    allocator, vkexec::buffer_create_info{ .size = k_buffer_size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT }));
  auto img = vkexec::test::sync_wait_value(vkexec::allocate_image(allocator,
    vkexec::image_create_info{
      .extent = { .width = k_width, .height = k_height, .depth = 1 },
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT,
      .mip_levels = 4,
      .array_layers = k_array_layers,
    }));
  auto cmd_result = ctx->allocate_command_buffer();
  REQUIRE(cmd_result.has_value());
  auto *cmd = vkexec::expected_take(cmd_result);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
  auto const family = ctx->queue_family();
  REQUIRE(vkexec::buffer_barrier(*ctx,
    cmd,
    { .buffer = buffer.handle(),
      .offset = k_barrier_offset,
      .size = k_barrier_size,
      .src_queue_family = family,
      .dst_queue_family = family }));
  REQUIRE(vkexec::image_barrier(*ctx,
    cmd,
    { .image = img.handle(),
      .range = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 2,
        .levelCount = 2,
        .baseArrayLayer = 3,
        .layerCount = 2 },
      .new_layout = VK_IMAGE_LAYOUT_GENERAL,
      .src_queue_family = family,
      .dst_queue_family = family }));
  REQUIRE(vkexec::image_barrier(*ctx,
    cmd,
    { .image = img.handle(),
      .range = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = VK_REMAINING_MIP_LEVELS,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS },
      .new_layout = VK_IMAGE_LAYOUT_GENERAL }));
  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
  REQUIRE(ctx->submit_and_wait(cmd));
  ctx->free_command_buffer(cmd);
}

TEST_CASE("resource barrier pass steps execute", "[vkexec][barrier][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto buffer = vkexec::test::sync_wait_value(vkexec::allocate_buffer(
    allocator, vkexec::buffer_create_info{ .size = k_buffer_size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT }));
  auto img = vkexec::test::sync_wait_value(vkexec::allocate_image(allocator,
    vkexec::image_create_info{ .extent = { .width = k_pass_extent, .height = k_pass_extent, .depth = 1 },
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT }));
  auto graph =
    stdexec::schedule(ctx->get_scheduler())
    | vkexec::barrier::buffer({ .buffer = buffer.handle(), .offset = k_barrier_offset, .size = k_barrier_size })
    | vkexec::barrier::image({ .image = img.handle(), .new_layout = VK_IMAGE_LAYOUT_GENERAL });
  auto waited = vkexec::test::sync_wait_sender(graph);
  REQUIRE(vkexec::test::sync_wait_completed(waited));
}
