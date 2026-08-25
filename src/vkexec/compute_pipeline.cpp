#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/config.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec_edsl/spirv.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace vkexec {

auto compute_pipeline::from_spirv(context &ctx, std::span<std::uint32_t const> spirv, layout_desc const &desc)
  -> compute_pipeline
{
  pipeline_resources &cached = ctx.get_or_create_from_spirv(spirv, desc);
  return compute_pipeline{ &ctx, &cached };
}

auto compute_pipeline::from_glsl(context &ctx, std::string_view glsl, layout_desc const &desc, std::string_view name)
  -> result<compute_pipeline>
{
  if (glsl.empty()) {
    return std::unexpected(make_error(errc::invalid_argument, "compute_pipeline::from_glsl requires non-empty GLSL"));
  }
  result<std::vector<std::uint32_t>> const spirv = edsl::compile_glsl_to_spirv(glsl, name, edsl::shader_kind::compute);
  if (!spirv) { return std::unexpected(spirv.error()); }
  return from_spirv(ctx, *spirv, desc);
}

auto compute_pipeline::allocate_set() -> VkDescriptorSet
{
  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = resources_->descriptor_pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &resources_->set_layout;
  VkDescriptorSet set{ VK_NULL_HANDLE };
  if (vkAllocateDescriptorSets(ctx_->device(), &dsai, &set) != VK_SUCCESS) {
    VKEXEC_THROW(std::runtime_error("vkAllocateDescriptorSets failed"));
  }
  return set;
}

auto compute_pipeline::update_set(VkDescriptorSet set, std::span<storage_binding const> buffers) -> void
{
  if (buffers.size() != resources_->binding_count) {
    VKEXEC_THROW(std::invalid_argument("update_set buffer count must match layout_desc.bindings"));
  }
  if (buffers.empty()) { return; }

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
}

}// namespace vkexec
