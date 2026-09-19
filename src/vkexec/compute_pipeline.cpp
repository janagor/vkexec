#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/compute_create.hpp>
#include <vkexec/detail/descriptor_backend.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/result.hpp>
#include <vkexec/spirv_compile.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vkexec {
namespace {

  [[nodiscard]] auto descriptor_type(resource_kind kind) noexcept -> VkDescriptorType
  {
    switch (kind) {
    case resource_kind::storage_buffer: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case resource_kind::storage_image: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case resource_kind::sampled_image: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case resource_kind::sampler: return VK_DESCRIPTOR_TYPE_SAMPLER;
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

auto destroy_compute_resources(context const &ctx, pipeline_resources &resources) noexcept -> void
{ detail::destroy_compute_resources_with<detail::set_descriptor_backend>(ctx, resources); }

auto create_compute_resources(context &ctx, std::span<std::uint32_t const> spirv, layout_desc const &desc)
  -> result<pipeline_resources>
{
  VKEXEC_TRY_ASSIGN(layout_bindings, make_compute_bindings(desc));
  detail::compute_create_info const info{ .bindings = layout_bindings,
    .push_bytes = desc.push_constant_size,
    .specialization = desc.specialization,
    .local_size = desc.local_size };
  return detail::create_compute_resources_with<detail::set_descriptor_backend>(
    ctx, spirv, info, "create_compute_resources requires non-empty SPIR-V");
}

auto create_compute_resources(context &ctx, std::string_view glsl, layout_desc const &desc, std::string_view name)
  -> result<pipeline_resources>
{
  if (glsl.empty()) { return fail(errc::invalid_argument, "create_compute_resources requires non-empty GLSL"); }
  VKEXEC_TRY_ASSIGN(spirv, compile_glsl_to_spirv(glsl, name, shader_kind::compute, ctx.api_version()));
  return create_compute_resources(ctx, spirv, desc);
}

auto allocate_compute_set(context const &ctx, pipeline_resources const &pipe) -> result<VkDescriptorSet>
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

auto bind_storage(context &ctx, pipeline_resources const &pipe, std::span<storage_binding const> buffers)
  -> result<bound_compute>
{
  if (buffers.size() != pipe.binding_count) {
    return fail(errc::invalid_argument, "bind_storage buffer count must match layout_desc.bindings");
  }
  VKEXEC_TRY_ASSIGN(set, allocate_compute_set(ctx, pipe));
  write_storage_descriptors(ctx.device(), set, buffers);
  return bound_compute{ .pipe = &pipe, .set = set };
}

auto free_compute_set(context const &ctx, pipeline_resources const &pipe, VkDescriptorSet set) noexcept -> void
{
  if (set == VK_NULL_HANDLE || pipe.descriptor_pool == VK_NULL_HANDLE) { return; }
  vkFreeDescriptorSets(ctx.device(), pipe.descriptor_pool, 1, &set);
}

auto compute_pipeline::reset() noexcept -> void
{
  if (ctx_ != nullptr && resources_ != nullptr) { destroy_compute_resources(*ctx_, *resources_); }
  resources_.reset();
  ctx_ = nullptr;
}

auto compute_pipeline::create(context &ctx, std::span<std::uint32_t const> spirv, layout_desc const &desc)
  -> detail::sync_sender_fn<compute_pipeline>
{
  return detail::make_sync_sender_fn<compute_pipeline>([&ctx, spirv, desc]() -> result<compute_pipeline> {
    VKEXEC_TRY_ASSIGN(owned, create_compute_resources(ctx, spirv, desc));
    return compute_pipeline{ &ctx, std::make_unique<pipeline_resources>(owned) };
  });
}

auto compute_pipeline::create(context &ctx, std::string_view glsl, layout_desc const &desc, std::string_view name)
  -> detail::sync_sender_fn<compute_pipeline>
{
  return detail::make_sync_sender_fn<compute_pipeline>(
    [&ctx, glsl = std::string(glsl), desc, name = std::string(name)]() -> result<compute_pipeline> {
      // Capture by value: the sender may outlive the caller's string_views.
      VKEXEC_TRY_ASSIGN(owned, create_compute_resources(ctx, glsl, desc, name));
      return compute_pipeline{ &ctx, std::make_unique<pipeline_resources>(owned) };
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
{ return vkexec::allocate_compute_set(*ctx_, *resources_); }

auto compute_pipeline::update_set(VkDescriptorSet set, std::span<storage_binding const> buffers) const -> status
{
  if (buffers.size() != resources_->binding_count) {
    return fail(errc::invalid_argument, "update_set buffer count must match layout_desc.bindings");
  }
  write_storage_descriptors(ctx_->device(), set, buffers);
  return {};
}

auto bind_storage_sender(compute_pipeline const &pipe, std::span<storage_binding const> buffers)
  -> detail::sync_sender_fn<bound_compute_pipeline>
{
  std::vector<storage_binding> owned(buffers.begin(), buffers.end());
  return detail::make_sync_sender_fn<bound_compute_pipeline>(
    [&pipe, owned = std::move(owned)]() mutable -> result<bound_compute_pipeline> {
      VKEXEC_TRY_ASSIGN(set, pipe.allocate_set());
      VKEXEC_TRY(pipe.update_set(set, owned));
      return bound_compute_pipeline{ .pipe = &pipe, .set = set };
    });
}

auto compute_pass(compute_pipeline const &pipe, VkDescriptorSet set, std::uint32_t work_count)
  -> prebuilt_compute_pass_closure
{ return compute_pass(pipe.bind(set), pipe.groups_for(work_count)); }

}// namespace vkexec
