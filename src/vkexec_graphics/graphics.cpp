#include <vkexec_graphics/graphics.hpp>

#include <vkexec/context.hpp>
#include <vkexec_edsl/trace.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <span>
#include <stdexcept>
#include <vector>

namespace vkexec {

auto graphics_pipeline::create_module(std::vector<std::uint32_t> const &spirv) const -> VkShaderModule
{
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = spirv.size() * sizeof(std::uint32_t);
  create_info.pCode = spirv.data();
  VkShaderModule module{ VK_NULL_HANDLE };
  if (vkCreateShaderModule(device_, &create_info, nullptr, &module) != VK_SUCCESS) {
    throw std::runtime_error("vkCreateShaderModule failed");
  }
  return module;
}

auto graphics_pipeline::complete([[maybe_unused]] context &ctx,
  VkRenderPass render_pass,
  std::vector<std::uint32_t> const &vs_spv,
  std::vector<std::uint32_t> const &fs_spv,
  std::span<edsl::storage_trace const> buffers) -> void
{
  buffers_.clear();
  buffers_.reserve(buffers.size());
  std::ranges::transform(
    buffers, std::back_inserter(buffers_), [](edsl::storage_trace const &buffer) -> bound_buffer {
      return bound_buffer{
        .binding = static_cast<std::uint32_t>(buffer.binding),
        .buffer = static_cast<VkBuffer>(buffer.vk_buffer),
        .byte_size = static_cast<VkDeviceSize>(buffer.byte_size),
      };
    });

  if (!buffers_.empty()) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(buffers_.size());
    for (std::size_t index = 0; index < buffers_.size(); ++index) {
      bindings.at(index).binding = buffers_.at(index).binding;
      bindings.at(index).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings.at(index).descriptorCount = 1;
      bindings.at(index).stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = static_cast<std::uint32_t>(bindings.size());
    dslci.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &dslci, nullptr, &set_layout_) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateDescriptorSetLayout failed (graphics)");
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = static_cast<std::uint32_t>(buffers_.size());
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(device_, &dpci, nullptr, &descriptor_pool_) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateDescriptorPool failed (graphics)");
    }

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = descriptor_pool_;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &set_layout_;
    if (vkAllocateDescriptorSets(device_, &dsai, &descriptor_set_) != VK_SUCCESS) {
      throw std::runtime_error("vkAllocateDescriptorSets failed (graphics)");
    }
  }

  VkShaderModule vert = create_module(vs_spv);
  VkShaderModule frag = create_module(fs_spv);

  static constexpr std::size_t k_graphics_stage_count = 2;
  // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
  std::array<VkPipelineShaderStageCreateInfo, k_graphics_stage_count> stages{};
  stages.at(0).sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages.at(0).stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages.at(0).module = vert;
  stages.at(0).pName = "main";
  stages.at(1).sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages.at(1).stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages.at(1).module = frag;
  stages.at(1).pName = "main";

  VkPipelineVertexInputStateCreateInfo vertex_input{};
  vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo input_assembly{};
  input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly.topology = cfg_.topology;

  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo raster{};
  raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = VK_CULL_MODE_NONE;
  raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  raster.lineWidth = 1.0F;

  // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
  VkPipelineMultisampleStateCreateInfo multisample{};
  multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState blend_attachment{};
  blend_attachment.colorWriteMask = static_cast<VkColorComponentFlags>(
    static_cast<std::uint32_t>(VK_COLOR_COMPONENT_R_BIT) | static_cast<std::uint32_t>(VK_COLOR_COMPONENT_G_BIT)
    | static_cast<std::uint32_t>(VK_COLOR_COMPONENT_B_BIT) | static_cast<std::uint32_t>(VK_COLOR_COMPONENT_A_BIT));
  if (cfg_.alpha_blend) {
    blend_attachment.blendEnable = VK_TRUE;
    blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
  }

  VkPipelineColorBlendStateCreateInfo blend{};
  blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  blend.attachmentCount = 1;
  blend.pAttachments = &blend_attachment;

  std::array dynamic_states{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
  VkPipelineDynamicStateCreateInfo dynamic{};
  dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
  dynamic.pDynamicStates = dynamic_states.data();

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  if (set_layout_ != VK_NULL_HANDLE) {
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &set_layout_;
  }
  if (vkCreatePipelineLayout(device_, &plci, nullptr, &layout_) != VK_SUCCESS) {
    vkDestroyShaderModule(device_, frag, nullptr);
    vkDestroyShaderModule(device_, vert, nullptr);
    throw std::runtime_error("vkCreatePipelineLayout failed");
  }

  VkGraphicsPipelineCreateInfo gpci{};
  gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  gpci.stageCount = static_cast<std::uint32_t>(stages.size());
  gpci.pStages = stages.data();
  gpci.pVertexInputState = &vertex_input;
  gpci.pInputAssemblyState = &input_assembly;
  gpci.pViewportState = &viewport_state;
  gpci.pRasterizationState = &raster;
  gpci.pMultisampleState = &multisample;
  gpci.pColorBlendState = &blend;
  gpci.pDynamicState = &dynamic;
  gpci.layout = layout_;
  gpci.renderPass = render_pass;
  gpci.subpass = 0;
  VkResult const created = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline_);
  vkDestroyShaderModule(device_, frag, nullptr);
  vkDestroyShaderModule(device_, vert, nullptr);
  if (created != VK_SUCCESS) { throw std::runtime_error("vkCreateGraphicsPipelines failed"); }
}

}// namespace vkexec
