#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/spirv_compile.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vkexec {

auto compute_pipeline::create(context &ctx, std::span<std::uint32_t const> spirv, layout_desc const &desc)
  -> detail::sync_sender_fn<compute_pipeline>
{
  return detail::make_sync_sender_fn<compute_pipeline>([&ctx, spirv, desc]() -> result<compute_pipeline> {
    VKEXEC_TRY_ASSIGN(cached, ctx.get_or_create_from_spirv(spirv, desc));
    return compute_pipeline{ &ctx, &cached.get() };
  });
}

auto compute_pipeline::create(context &ctx, std::string_view glsl, layout_desc const &desc, std::string_view name)
  -> detail::sync_sender_fn<compute_pipeline>
{
  return detail::make_sync_sender_fn<compute_pipeline>(
    [&ctx, glsl = std::string(glsl), desc, name = std::string(name)]() -> result<compute_pipeline> {
      // Capture by value: the sender may outlive the caller's string_views.
      if (glsl.empty()) { return fail(errc::invalid_argument, "compute_pipeline::create requires non-empty GLSL"); }
      VKEXEC_TRY_ASSIGN(spirv, compile_glsl_to_spirv(glsl, name, shader_kind::compute, ctx.api_version()));
      VKEXEC_TRY_ASSIGN(cached, ctx.get_or_create_from_spirv(spirv, desc));
      return compute_pipeline{ &ctx, &cached.get() };
    });
}

auto compute_pipeline::allocate_set_sender() const -> detail::sync_sender_fn<VkDescriptorSet>
{
  return detail::make_sync_sender_fn<VkDescriptorSet>([this]() -> result<VkDescriptorSet> { return allocate_set(); });
}

auto compute_pipeline::update_set_sender(VkDescriptorSet set, std::span<storage_binding const> buffers) const
  -> detail::sync_void_sender_fn
{
  std::vector<storage_binding> owned(buffers.begin(), buffers.end());
  return detail::make_sync_void_sender_fn(
    [this, set, owned = std::move(owned)]() -> status { return update_set(set, owned); });
}

auto compute_pipeline::allocate_set() const -> result<VkDescriptorSet>
{
  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = resources_->descriptor_pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &resources_->set_layout;
  VkDescriptorSet set{ VK_NULL_HANDLE };
  VkResult const allocate_result = vkAllocateDescriptorSets(ctx_->device(), &dsai, &set);
  if (allocate_result != VK_SUCCESS) { return fail(allocate_result, "vkAllocateDescriptorSets failed"); }
  return set;
}

auto compute_pipeline::update_set(VkDescriptorSet set, std::span<storage_binding const> buffers) const -> status
{
  if (buffers.size() != resources_->binding_count) {
    return fail(errc::invalid_argument, "update_set buffer count must match layout_desc.bindings");
  }
  if (buffers.empty()) { return {}; }

  std::vector<VkDescriptorBufferInfo> infos(buffers.size());
  std::vector<VkWriteDescriptorSet> writes(buffers.size());
  std::size_t index = 0;
  for (storage_binding const &buffer : buffers) {
    infos.at(index).buffer = buffer.buffer;
    infos.at(index).offset = 0;
    infos.at(index).range = buffer.byte_size;
    writes.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes.at(index).dstSet = set;
    writes.at(index).dstBinding = static_cast<std::uint32_t>(index);
    writes.at(index).descriptorCount = 1;
    writes.at(index).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes.at(index).pBufferInfo = &infos.at(index);
    ++index;
  }
  vkUpdateDescriptorSets(ctx_->device(), static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
  return {};
}

auto bind_storage_sender(compute_pipeline const &pipe, std::span<storage_binding const> buffers)
  -> detail::sync_sender_fn<bound_compute_pipeline>
{
  std::vector<storage_binding> owned(buffers.begin(), buffers.end());
  return detail::make_sync_sender_fn<bound_compute_pipeline>(
    [pipe, owned = std::move(owned)]() mutable -> result<bound_compute_pipeline> {
      VKEXEC_TRY_ASSIGN(set, pipe.allocate_set());
      VKEXEC_TRY(pipe.update_set(set, owned));
      return bound_compute_pipeline{ .pipe = pipe, .set = set };
    });
}

auto compute_pass(compute_pipeline const &pipe, VkDescriptorSet set, std::uint32_t work_count)
  -> prebuilt_compute_pass_closure
{ return compute_pass(pipe.bind(set), pipe.groups_for(work_count)); }

}// namespace vkexec
