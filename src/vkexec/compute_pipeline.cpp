#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/compute_create.hpp>
#include <vkexec/detail/descriptor_backend.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vkexec {
namespace {

  [[nodiscard]] auto descriptor_type(resource_kind kind) noexcept -> VkDescriptorType
  {
    switch (kind) {
    case resource_kind::storage_buffer:
      return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case resource_kind::storage_image:
      return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case resource_kind::sampled_image:
      return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case resource_kind::sampler:
      return VK_DESCRIPTOR_TYPE_SAMPLER;
    }
    return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  }

  auto make_compute_bindings(layout_desc const &desc) -> result<std::vector<VkDescriptorSetLayoutBinding>>
  {
    if (!desc.binding_slots.empty() && desc.binding_slots.size() != desc.bindings.size()) {
      return fail(errc::invalid_argument, "layout_desc binding_slots must match bindings");
    }
    if (!desc.binding_kinds.empty() && desc.binding_kinds.size() != desc.bindings.size()) {
      return fail(errc::invalid_argument, "layout_desc binding_kinds must match bindings");
    }
    std::vector<VkDescriptorSetLayoutBinding> layout_bindings(desc.bindings.size());
    for (std::size_t index = 0; index < desc.bindings.size(); ++index) {
      layout_bindings.at(index).binding =
        desc.binding_slots.empty() ? static_cast<std::uint32_t>(index) : desc.binding_slots.at(index);
      layout_bindings.at(index).descriptorType =
        descriptor_type(desc.binding_kinds.empty() ? resource_kind::storage_buffer : desc.binding_kinds.at(index));
      layout_bindings.at(index).descriptorCount = 1;
      layout_bindings.at(index).stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    return layout_bindings;
  }

}// namespace

auto destroy(context const &ctx, handles::compute_pipeline &resources) noexcept -> void
{ detail::destroy_compute_resources_with<detail::set_descriptor_backend>(ctx, resources); }

auto create(context &ctx, std::span<std::uint32_t const> spirv, layout_desc const &desc)
  -> result<handles::compute_pipeline>
{
  VKEXEC_TRY_ASSIGN(layout_bindings, make_compute_bindings(desc));
  detail::compute_create_info const info{ .bindings = layout_bindings,
    .push_bytes = desc.push_constant_size,
    .specialization = desc.specialization,
    .local_size = desc.local_size };
  return detail::create_compute_resources_with<detail::set_descriptor_backend>(
    ctx, spirv, info, "create requires non-empty SPIR-V");
}

auto allocate_compute_set(context const &ctx, handles::compute_pipeline const &pipe) -> result<VkDescriptorSet>
{
  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = pipe.descriptor_pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &pipe.set_layout;
  VkDescriptorSet set{ VK_NULL_HANDLE };
  VkResult const allocate_result = vkAllocateDescriptorSets(ctx.device(), &dsai, &set);
  if (allocate_result != VK_SUCCESS) { return fail(allocate_result, "vkAllocateDescriptorSets failed"); }
  return set;
}

auto bind_storage(context &ctx, handles::compute_pipeline const &pipe, std::span<storage_binding const> buffers)
  -> result<bound_compute>
{
  if (buffers.size() != pipe.binding_count) {
    return fail(errc::invalid_argument, "bind_storage buffer count must match layout_desc.bindings");
  }
  VKEXEC_TRY_ASSIGN(set, allocate_compute_set(ctx, pipe));
  write_storage_descriptors(ctx.device(), set, buffers);
  return bound_compute{ .pipe = &pipe, .set = set };
}

auto free_compute_set(context const &ctx, handles::compute_pipeline const &pipe, VkDescriptorSet set) noexcept -> void
{
  if (set == VK_NULL_HANDLE || pipe.descriptor_pool == VK_NULL_HANDLE) { return; }
  vkFreeDescriptorSets(ctx.device(), pipe.descriptor_pool, 1, &set);
}

auto owned::compute_pipeline::reset() noexcept -> void
{
  if (ctx_ != nullptr && resources_ != nullptr) { destroy(*ctx_, *resources_); }
  resources_.reset();
  ctx_ = nullptr;
}

auto detail::make_compute_pipeline_spirv_factory::operator()() const -> result<::vkexec::owned::compute_pipeline>
{
  VKEXEC_TRY_ASSIGN(owned, create(*ctx, spirv, desc));
  return ::vkexec::owned::compute_pipeline::make(*ctx, std::make_unique<handles::compute_pipeline>(owned));
}

auto detail::allocate_set_factory::operator()() const -> result<VkDescriptorSet> { return pipe->allocate_set(); }

auto detail::update_set_factory::operator()() const -> status { return pipe->update_set(set, buffers); }

auto detail::bind_storage_factory::operator()() const -> result<bound_compute_pipeline>
{
  VKEXEC_TRY_ASSIGN(set, pipe->allocate_set());
  VKEXEC_TRY(pipe->update_set(set, buffers));
  return bound_compute_pipeline{ .pipe = pipe, .set = set };
}

auto owned::compute_pipeline::allocate_set() const -> result<VkDescriptorSet>
{ return vkexec::allocate_compute_set(*ctx_, *resources_); }

auto owned::compute_pipeline::update_set(VkDescriptorSet set, std::span<storage_binding const> buffers) const -> status
{
  if (buffers.size() != resources_->binding_count) {
    return fail(errc::invalid_argument, "update_set buffer count must match layout_desc.bindings");
  }
  write_storage_descriptors(ctx_->device(), set, buffers);
  return {};
}

}// namespace vkexec
