// NOLINTBEGIN(readability-function-cognitive-complexity)

#include "strict_gpu.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/barrier.hpp>
#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/resource_use.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

namespace {

constexpr VkDeviceSize k_buffer_size = 512;
constexpr VkDeviceSize k_half_size = 256;
constexpr VkDeviceSize k_snapshot_begin = 128;
constexpr VkDeviceSize k_overwrite_begin = 192;
constexpr VkDeviceSize k_overwrite_size = 128;
constexpr std::uint32_t k_first_pattern = 0x12345678U;
constexpr std::uint32_t k_second_pattern = 0xABCDEF01U;
constexpr std::uint32_t k_overwrite_pattern = 0x87654321U;
constexpr std::uint32_t k_copy_count = 1;
[[nodiscard]] auto host_read_barrier(VkBuffer buffer) -> decltype(auto)
{
  return vkexec::barrier::buffer({ .buffer = buffer,
    .src_stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
    .dst_stage = VK_PIPELINE_STAGE_2_HOST_BIT,
    .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
    .dst_access = VK_ACCESS_2_HOST_READ_BIT });
}

}// namespace

TEST_CASE("buffer suballocations preserve data across partial graph overlaps", "[mutation-gpu][gpu-overlap]")
{
  auto ctx = mutation_gpu::context();
  auto allocator = mutation_gpu::wait(vkexec::vma::factory::make_allocator(*ctx));
  auto source = mutation_gpu::wait(vkexec::allocate_buffer(allocator,
    vkexec::buffer_create_info{ .size = k_buffer_size,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .memory = vkexec::memory_domain::device_local }));
  auto readback = mutation_gpu::wait(vkexec::allocate_buffer(allocator,
    vkexec::buffer_create_info{ .size = k_half_size + k_buffer_size,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .memory = vkexec::memory_domain::staging }));
  auto *const buffer = source.handle();
  auto *const output = readback.handle();
  auto graph =
    stdexec::schedule(ctx->get_scheduler())
    | vkexec::custom_pass(
      vkexec::uses(vkexec::buffer_extent{ .buffer = buffer, .size = k_buffer_size },
        vkexec::buffer_extent{ .buffer = output, .size = k_half_size + k_buffer_size },
        vkexec::write(buffer, 0, k_half_size, vkexec::buffer_usage::transfer_destination),
        vkexec::write(buffer, k_half_size, k_half_size, vkexec::buffer_usage::transfer_destination)),
      [buffer](VkCommandBuffer cmd) -> void {
        vkCmdFillBuffer(cmd, buffer, 0, k_half_size, k_first_pattern);
        vkCmdFillBuffer(cmd, buffer, k_half_size, k_half_size, k_second_pattern);
      })
    | vkexec::custom_pass(
      vkexec::uses(vkexec::read(buffer, k_snapshot_begin, k_half_size, vkexec::buffer_usage::transfer_source),
        vkexec::write(output, 0, k_half_size, vkexec::buffer_usage::transfer_destination)),
      [buffer, output](VkCommandBuffer cmd) -> void {
        VkBufferCopy const copy{ .srcOffset = k_snapshot_begin, .dstOffset = 0, .size = k_half_size };
        vkCmdCopyBuffer(cmd, buffer, output, k_copy_count, &copy);
      })
    | vkexec::custom_pass(vkexec::uses(vkexec::write(
                            buffer, k_overwrite_begin, k_overwrite_size, vkexec::buffer_usage::transfer_destination)),
      [buffer](VkCommandBuffer cmd) -> void {
        vkCmdFillBuffer(cmd, buffer, k_overwrite_begin, k_overwrite_size, k_overwrite_pattern);
      })
    | vkexec::custom_pass(
      vkexec::uses(vkexec::read(buffer, vkexec::buffer_usage::transfer_source),
        vkexec::write(output, k_half_size, k_buffer_size, vkexec::buffer_usage::transfer_destination)),
      [buffer, output](VkCommandBuffer cmd) -> void {
        VkBufferCopy const copy{ .srcOffset = 0, .dstOffset = k_half_size, .size = k_buffer_size };
        vkCmdCopyBuffer(cmd, buffer, output, k_copy_count, &copy);
      })
    | host_read_barrier(output);
  auto const waited = vkexec::try_sync_wait(std::move(graph));
  if (waited.failed() || waited.stopped || !waited.values) { mutation_gpu::infrastructure("submission failed"); }
  if (!readback.invalidate()) { mutation_gpu::infrastructure("readback invalidation failed"); }
  auto const bytes = readback.mapped();
  if (bytes.size() != k_half_size + k_buffer_size) { mutation_gpu::infrastructure("incomplete readback mapping"); }
  for (VkDeviceSize offset{}; offset < readback.size(); offset += sizeof(std::uint32_t)) {
    std::uint32_t actual{};
    auto const word = bytes.subspan(offset, sizeof(actual));
    std::memcpy(&actual, word.data(), sizeof(actual));
    bool const snapshot = offset < k_half_size;
    auto const source_offset = snapshot ? offset + k_snapshot_begin : offset - k_half_size;
    auto const expected = [snapshot, source_offset]() -> std::uint32_t {
      if (!snapshot && source_offset >= k_overwrite_begin && source_offset < k_overwrite_begin + k_overwrite_size) {
        return k_overwrite_pattern;
      }
      if (source_offset < k_half_size) { return k_first_pattern; }
      return k_second_pattern;
    }();
    REQUIRE(actual == expected);
  }
}

TEST_CASE("disjoint buffer declarations preserve both regions", "[mutation-gpu][gpu-disjoint]")
{
  auto ctx = mutation_gpu::context();
  auto allocator = mutation_gpu::wait(vkexec::vma::factory::make_allocator(*ctx));
  auto source = mutation_gpu::wait(vkexec::allocate_buffer(allocator,
    vkexec::buffer_create_info{ .size = k_buffer_size,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .memory = vkexec::memory_domain::device_local }));
  auto readback = mutation_gpu::wait(vkexec::allocate_buffer(allocator,
    vkexec::buffer_create_info{
      .size = k_buffer_size, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .memory = vkexec::memory_domain::staging }));
  auto buffer = source.handle();
  auto output = readback.handle();
  auto graph =
    stdexec::schedule(ctx->get_scheduler())
    | vkexec::custom_pass(vkexec::uses(vkexec::buffer_extent{ .buffer = buffer, .size = k_buffer_size },
                            vkexec::buffer_extent{ .buffer = output, .size = k_buffer_size },
                            vkexec::write(buffer, 0, k_half_size, vkexec::buffer_usage::transfer_destination)),
      [buffer](VkCommandBuffer cmd) { vkCmdFillBuffer(cmd, buffer, 0, k_half_size, k_first_pattern); })
    | vkexec::custom_pass(
      vkexec::uses(vkexec::write(buffer, k_half_size, k_half_size, vkexec::buffer_usage::transfer_destination)),
      [buffer](VkCommandBuffer cmd) { vkCmdFillBuffer(cmd, buffer, k_half_size, k_half_size, k_second_pattern); })
    | vkexec::custom_pass(vkexec::uses(vkexec::read(buffer, 0, k_half_size, vkexec::buffer_usage::transfer_source),
                            vkexec::read(buffer, k_half_size, k_half_size, vkexec::buffer_usage::transfer_source),
                            vkexec::write(output, vkexec::buffer_usage::transfer_destination)),
      [buffer, output](VkCommandBuffer cmd) {
        VkBufferCopy const copy{ .srcOffset = 0, .dstOffset = 0, .size = k_buffer_size };
        vkCmdCopyBuffer(cmd, buffer, output, k_copy_count, &copy);
      })
    | host_read_barrier(output);
  auto const waited = vkexec::try_sync_wait(std::move(graph));
  if (waited.failed() || waited.stopped || !waited.values) { mutation_gpu::infrastructure("submission failed"); }
  if (!readback.invalidate()) { mutation_gpu::infrastructure("readback invalidation failed"); }
  auto bytes = readback.mapped();
  if (bytes.size() != k_buffer_size) { mutation_gpu::infrastructure("incomplete readback mapping"); }
  for (VkDeviceSize offset{}; offset < k_buffer_size; offset += sizeof(std::uint32_t)) {
    std::uint32_t actual{};
    std::memcpy(&actual, bytes.subspan(offset, sizeof(actual)).data(), sizeof(actual));
    REQUIRE(actual == (offset < k_half_size ? k_first_pattern : k_second_pattern));
  }
}
// NOLINTEND(readability-function-cognitive-complexity)
