#ifndef VKEXEC_DETAIL_DESCRIPTOR_BACKEND_HPP
#define VKEXEC_DETAIL_DESCRIPTOR_BACKEND_HPP

#include <vkexec/context.hpp>
#include <vkexec/detail/descriptor_table_backend.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vkexec::detail {

inline constexpr std::uint32_t k_descriptor_sets_per_pool = 64;

struct descriptor_layout_info
{
  std::span<VkDescriptorSetLayoutBinding const> bindings;
  VkShaderStageFlags push_stages{ 0 };
  std::uint32_t push_bytes{ 0 };
  std::uint32_t sets_per_pool{ k_descriptor_sets_per_pool };
  bool create_set_layout{ true };
  bool create_pool{ true };
};

template<class Backend>
concept descriptor_backend = requires(context const *ctx,
  VkDevice device,
  VkCommandBuffer cmd,
  compute_bind const &bind,
  handles::compute_pipeline &resources,
  descriptor_layout_info const &layout_info,
  std::span<std::byte const> push,
  VkPipelineBindPoint bind_point) {
  { Backend::pipeline_create_flags() } -> std::convertible_to<VkPipelineCreateFlags2>;
  { Backend::create_set_and_pipeline_layout(device, resources, layout_info) } -> std::same_as<status>;
  { Backend::create_descriptor_pool(device, resources, layout_info) } -> std::same_as<status>;
  { Backend::destroy_binding_objects(device, resources) } -> std::same_as<void>;
  { Backend::bind_resources(cmd, bind_point, bind) } -> std::same_as<void>;
  { Backend::push(ctx, cmd, bind_point, bind, push) } -> std::same_as<status>;
};

struct set_descriptor_backend
{
  using bound_type = VkDescriptorSet;
  using lower_env = empty_table_lower_env;

  [[nodiscard]] static constexpr auto pipeline_create_flags() noexcept -> VkPipelineCreateFlags2 { return 0; }

  template<class Resources>
  [[nodiscard]] static auto
    create_set_and_pipeline_layout(VkDevice device, Resources &resources, descriptor_layout_info const &info) -> status
  {
    if (info.create_set_layout) {
      VkDescriptorSetLayoutCreateInfo set_info{};
      set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      set_info.bindingCount = static_cast<std::uint32_t>(info.bindings.size());
      set_info.pBindings = info.bindings.data();
      if (VkResult const result = vkCreateDescriptorSetLayout(device, &set_info, nullptr, &resources.set_layout);
        result != VK_SUCCESS) {
        return fail(result, "vkCreateDescriptorSetLayout failed");
      }
    }

    VkPushConstantRange push_range{};
    push_range.stageFlags = info.push_stages;
    push_range.size = info.push_bytes;
    VkPipelineLayoutCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    if (resources.set_layout != VK_NULL_HANDLE) {
      pipeline_info.setLayoutCount = 1;
      pipeline_info.pSetLayouts = &resources.set_layout;
    }
    if (info.push_bytes > 0) {
      pipeline_info.pushConstantRangeCount = 1;
      pipeline_info.pPushConstantRanges = &push_range;
    }
    if (VkResult const result = vkCreatePipelineLayout(device, &pipeline_info, nullptr, &resources.pipeline_layout);
      result != VK_SUCCESS) {
      return fail(result, "vkCreatePipelineLayout failed");
    }
    return {};
  }

  template<class Resources>
  [[nodiscard]] static auto
    create_descriptor_pool(VkDevice device, Resources &resources, descriptor_layout_info const &info) -> status
  {
    if (!info.create_pool) { return {}; }
    std::vector<VkDescriptorPoolSize> pool_sizes;
    for (VkDescriptorSetLayoutBinding const &binding : info.bindings) {
      auto const found = std::ranges::find_if(pool_sizes,
        [&binding](VkDescriptorPoolSize const &size) -> bool { return size.type == binding.descriptorType; });
      std::uint32_t const count = binding.descriptorCount * info.sets_per_pool;
      if (found == pool_sizes.end()) {
        pool_sizes.push_back(VkDescriptorPoolSize{ .type = binding.descriptorType, .descriptorCount = count });
      } else {
        found->descriptorCount += count;
      }
    }
    if (pool_sizes.empty()) {
      pool_sizes.push_back(
        VkDescriptorPoolSize{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = info.sets_per_pool });
    }
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = info.sets_per_pool;
    pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    if (VkResult const result = vkCreateDescriptorPool(device, &pool_info, nullptr, &resources.descriptor_pool);
      result != VK_SUCCESS) {
      return fail(result, "vkCreateDescriptorPool failed");
    }
    return {};
  }

  template<class Resources> static auto destroy_binding_objects(VkDevice device, Resources &resources) -> void
  {
    if (resources.pipeline_layout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device, resources.pipeline_layout, nullptr);
    }
    if (resources.descriptor_pool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device, resources.descriptor_pool, nullptr);
    }
    if (resources.set_layout != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device, resources.set_layout, nullptr); }
    resources.pipeline_layout = VK_NULL_HANDLE;
    resources.descriptor_pool = VK_NULL_HANDLE;
    resources.set_layout = VK_NULL_HANDLE;
  }

  template<class Bind>
  static auto bind_resources(VkCommandBuffer cmd, VkPipelineBindPoint bind_point, Bind const &bind) -> void
  {
    if (bind.set != VK_NULL_HANDLE) {
      vkCmdBindDescriptorSets(cmd, bind_point, bind.layout, 0, 1, &bind.set, 0, nullptr);
    }
  }

  template<class Bind>
  [[nodiscard]] static auto push(context const * /*ctx*/,
    VkCommandBuffer cmd,
    VkPipelineBindPoint bind_point,
    Bind const &bind,
    std::span<std::byte const> bytes) -> status
  {
    if (bind.layout != VK_NULL_HANDLE && !bytes.empty()) {
      VkShaderStageFlags const stages =
        bind_point == VK_PIPELINE_BIND_POINT_COMPUTE
          ? static_cast<VkShaderStageFlags>(VK_SHADER_STAGE_COMPUTE_BIT)
          : static_cast<VkShaderStageFlags>(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
      vkCmdPushConstants(cmd, bind.layout, stages, 0, static_cast<std::uint32_t>(bytes.size()), bytes.data());
    }
    return {};
  }

  [[nodiscard]] static auto
    lower(context &ctx, handles::compute_pipeline const &pipe, resource_table const &table, lower_env const & /*env*/)
      -> result<bound_type>
  {
    if (table.size() != pipe.binding_count) {
      return fail(errc::invalid_argument, "resource_table size must match pipeline binding count");
    }
    auto allocated = allocate_compute_set(ctx, pipe);
    if (!allocated) { return fail(allocated); }
    bound_type const set = expected_take(allocated);
    write_resource_descriptors(ctx.device(), set, table.entries());
    return set;
  }

  [[nodiscard]] static auto make_bind(handles::compute_pipeline const &pipe, bound_type bound) -> compute_bind
  { return bind_compute(pipe, bound); }

  static auto release(context &ctx, handles::compute_pipeline const &pipe, bound_type bound) -> void
  { free_compute_set(ctx, pipe, bound); }
};

static_assert(descriptor_backend<set_descriptor_backend>);
static_assert(descriptor_table_backend<set_descriptor_backend>);

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_DESCRIPTOR_BACKEND_HPP
