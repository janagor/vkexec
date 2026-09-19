#include <vkexec/submit_scope.hpp>

#include <vkexec/context.hpp>
#include <vkexec/detail/result.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace vkexec {

auto write_storage_descriptors(VkDevice device, VkDescriptorSet set, std::span<storage_binding const> buffers) -> void
{
  if (buffers.empty()) { return; }
  std::vector<VkDescriptorBufferInfo> buf_infos(buffers.size());
  std::vector<VkWriteDescriptorSet> writes(buffers.size());
  std::size_t index = 0;
  for (storage_binding const &buffer : buffers) {
    buf_infos.at(index).buffer = buffer.buffer;
    buf_infos.at(index).offset = 0;
    buf_infos.at(index).range = buffer.byte_size;
    writes.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes.at(index).dstSet = set;
    writes.at(index).dstBinding = buffer.binding;
    writes.at(index).descriptorCount = 1;
    writes.at(index).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes.at(index).pBufferInfo = &buf_infos.at(index);
    ++index;
  }
  vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

auto write_resource_descriptors(
  VkDevice device, VkDescriptorSet set, std::span<resource_binding const> resources) -> void
{
  if (resources.empty()) { return; }
  std::vector<VkDescriptorBufferInfo> buffer_infos(resources.size());
  std::vector<VkDescriptorImageInfo> image_infos(resources.size());
  std::vector<VkWriteDescriptorSet> writes(resources.size());
  std::size_t index = 0;
  for (resource_binding const &binding : resources) {
    VkWriteDescriptorSet &write = writes.at(index);
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding.slot;
    write.descriptorCount = 1;
    switch (binding.resource.kind) {
    case resource_kind::storage_buffer:
      buffer_infos.at(index).buffer = binding.resource.buffer;
      buffer_infos.at(index).range = binding.resource.byte_size;
      write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      write.pBufferInfo = &buffer_infos.at(index);
      break;
    case resource_kind::storage_image:
      image_infos.at(index).imageView = binding.resource.image_view;
      image_infos.at(index).imageLayout = binding.resource.image_layout;
      write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      write.pImageInfo = &image_infos.at(index);
      break;
    case resource_kind::sampled_image:
      image_infos.at(index).imageView = binding.resource.image_view;
      image_infos.at(index).imageLayout = binding.resource.image_layout;
      write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      write.pImageInfo = &image_infos.at(index);
      break;
    case resource_kind::sampler:
      image_infos.at(index).sampler = binding.resource.sampler;
      write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
      write.pImageInfo = &image_infos.at(index);
      break;
    }
    ++index;
  }
  vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

namespace detail {

  auto descriptor_cleanup::release(context const &ctx) noexcept -> void
  {
    // Host lock matches allocate path; pool frees must be serialized with submits.
    std::unique_lock const lock = ctx.lock_host();
    for (allocated_set const &item : allocated) { vkFreeDescriptorSets(ctx.device(), item.pool, 1, &item.set); }
    allocated.clear();
    sets.clear();
  }

  auto storage_bindings_equal(std::span<storage_binding const> lhs, std::span<storage_binding const> rhs) -> bool
  {
    return lhs.size() == rhs.size()
           && std::equal(lhs.begin(),
             lhs.end(),
             rhs.begin(),
             [](storage_binding const &left, storage_binding const &right) -> bool {
               return left.buffer == right.buffer && left.byte_size == right.byte_size && left.binding == right.binding;
             });
  }

  auto allocate_compute_set(context const &ctx, pipeline_resources &pipe, std::span<storage_binding const> buffers)
    -> detail::result<VkDescriptorSet>
  {
    std::unique_lock const lock = ctx.lock_host();
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = pipe.descriptor_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &pipe.set_layout;
    VkDescriptorSet set{ VK_NULL_HANDLE };
    if (VkResult const result = vkAllocateDescriptorSets(ctx.device(), &dsai, &set); result != VK_SUCCESS) {
      return detail::fail(result, "vkAllocateDescriptorSets failed");
    }
    write_storage_descriptors(ctx.device(), set, buffers);
    return set;
  }

  auto bind_or_allocate_set(context const &ctx,
    pipeline_resources &pipe,
    std::span<storage_binding const> buffers,
    descriptor_cleanup &cleanup) -> detail::result<VkDescriptorSet>
  {
    // Reuse a set already allocated for this pipeline in the same submit when bindings match.
    if (auto found = cleanup.sets.find(&pipe); found != cleanup.sets.end()) {
      if (storage_bindings_equal(found->second.buffers, buffers)) { return found->second.set; }
    }

    auto set = allocate_compute_set(ctx, pipe, buffers);
    if (!set) { return detail::fail(set); }
    auto *allocated = detail::expected_take(set);
    cleanup.sets.insert_or_assign(
      &pipe, descriptor_cleanup::pipeline_set_entry{ .buffers = { buffers.begin(), buffers.end() }, .set = allocated });
    cleanup.track(pipe.descriptor_pool, allocated);
    return allocated;
  }

  auto submit_scope::open(context &host) -> detail::result<submit_scope>
  {
    auto cmd = host.allocate_command_buffer();
    if (!cmd) { return detail::fail(cmd); }
    auto *cmd_buf = detail::expected_take(cmd);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (VkResult const result = vkBeginCommandBuffer(cmd_buf, &begin); result != VK_SUCCESS) {
      host.free_command_buffer(cmd_buf);
      return detail::fail(result, "vkBeginCommandBuffer failed");
    }

    submit_scope scope;
    scope.ctx = &host;
    scope.cmd = cmd_buf;
    return scope;
  }

  // NOLINTNEXTLINE(readability-make-member-function-const) -- ends Vulkan recording; not logically const
  auto submit_scope::end_recording() -> detail::status
  {
    if (cmd == VK_NULL_HANDLE) { return detail::fail(errc::invalid_argument, "submit_scope has no command buffer"); }
    if (VkResult const result = vkEndCommandBuffer(cmd); result != VK_SUCCESS) {
      return detail::fail(result, "vkEndCommandBuffer failed");
    }
    return {};
  }

  auto submit_scope::release() noexcept -> void
  {
    // Always free cmd before descriptors so a failed submit still returns loans.
    if (ctx == nullptr) { return; }
    if (cmd != VK_NULL_HANDLE) {
      ctx->free_command_buffer(cmd);
      cmd = VK_NULL_HANDLE;
    }
    cleanup.release(*ctx);
    ctx = nullptr;
  }

  // Same reclaim order as completion_waiter: wait, then destroy owned sync objects.
  auto reclaim_submission_sync(VkDevice device, VkQueue fallback_queue, VkSemaphore semaphore, VkFence fence) noexcept
    -> void
  {
    if (fence != VK_NULL_HANDLE) {
      (void)vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    } else if (fallback_queue != VK_NULL_HANDLE) {
      (void)vkQueueWaitIdle(fallback_queue);
    }
    if (semaphore != VK_NULL_HANDLE) { vkDestroySemaphore(device, semaphore, nullptr); }
    if (fence != VK_NULL_HANDLE) { vkDestroyFence(device, fence, nullptr); }
  }

  auto enter_submit_scope(context *ctx) -> enter_submit_scope_sender { return enter_submit_scope_sender{ .ctx = ctx }; }

  auto enter_submit_scope(context &ctx) -> enter_submit_scope_sender { return enter_submit_scope(&ctx); }

  auto submit_and_wait(submit_scope scope) -> submit_and_wait_sender
  { return submit_and_wait_sender{ .scope = std::move(scope) }; }

  auto submit_fence(submit_scope scope) -> submit_fence_sender
  { return submit_fence_sender{ .scope = std::move(scope) }; }

}// namespace detail
}// namespace vkexec
