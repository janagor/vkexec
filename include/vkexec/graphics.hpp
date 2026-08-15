#pragma once

#include <vkexec/context.hpp>
#include <vkexec/detail/glsl_emit.hpp>
#include <vkexec/detail/spirv.hpp>
#include <vkexec/detail/types.hpp>

#include <vulkan/vulkan.h>

#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vkexec {

/// Graphics pipeline built by tracing vertex/fragment eDSL lambdas to GLSL/SPIR-V.
class graphics_pipeline {
public:
  template<typename VertexFn, typename FragmentFn>
  graphics_pipeline(context &ctx, VkRenderPass render_pass, VertexFn &&vertex_fn, FragmentFn &&fragment_fn)
    : device_(ctx.device())
  {
    vlk::ASTContext vs_ast;
    {
      vlk::ASTScope scope(vs_ast);
      vlk::Int vid = vlk::Int::vertex_index();
      vlk::VertexWriter out;
      std::forward<VertexFn>(vertex_fn)(vid, out);
    }
    const std::string vs_glsl = detail::emit_vertex_glsl(vs_ast);
    const auto vs_spv = compile_glsl_to_spirv(vs_glsl, "vkexec.vert", shader_kind::vertex);

    vlk::ASTContext fs_ast;
    {
      vlk::ASTScope scope(fs_ast);
      vlk::FragmentReader in;
      vlk::FragmentWriter out;
      std::forward<FragmentFn>(fragment_fn)(in, out);
    }
    const std::string fs_glsl = detail::emit_fragment_glsl(fs_ast);
    const auto fs_spv = compile_glsl_to_spirv(fs_glsl, "vkexec.frag", shader_kind::fragment);

    VkShaderModule vert = create_module(vs_spv);
    VkShaderModule frag = create_module(fs_spv);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

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
    if (vkCreatePipelineLayout(device_, &plci, nullptr, &layout_) != VK_SUCCESS) {
      vkDestroyShaderModule(device_, frag, nullptr);
      vkDestroyShaderModule(device_, vert, nullptr);
      throw std::runtime_error("vkCreatePipelineLayout failed");
    }

    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.stageCount = 2;
    gpci.pStages = stages;
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
    const VkResult created = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline_);
    vkDestroyShaderModule(device_, frag, nullptr);
    vkDestroyShaderModule(device_, vert, nullptr);
    if (created != VK_SUCCESS) { throw std::runtime_error("vkCreateGraphicsPipelines failed"); }
  }

  ~graphics_pipeline()
  {
    if (device_ == VK_NULL_HANDLE) { return; }
    if (pipeline_ != VK_NULL_HANDLE) { vkDestroyPipeline(device_, pipeline_, nullptr); }
    if (layout_ != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device_, layout_, nullptr); }
  }

  graphics_pipeline(const graphics_pipeline &) = delete;
  graphics_pipeline &operator=(const graphics_pipeline &) = delete;

  graphics_pipeline(graphics_pipeline &&other) noexcept
    : device_(other.device_), layout_(other.layout_), pipeline_(other.pipeline_)
  {
    other.device_ = VK_NULL_HANDLE;
    other.layout_ = VK_NULL_HANDLE;
    other.pipeline_ = VK_NULL_HANDLE;
  }

  [[nodiscard]] VkPipeline pipeline() const noexcept { return pipeline_; }

  /// Begin the render pass, bind this pipeline, draw `vertex_count` verts, end the pass, and end the cmd buffer.
  void draw(VkCommandBuffer cmd,
    VkRenderPass render_pass,
    VkFramebuffer framebuffer,
    VkExtent2D extent,
    std::uint32_t vertex_count) const
  {
    VkClearValue clear{};
    clear.color = { { 0.08f, 0.09f, 0.12f, 1.0f } };

    VkRenderPassBeginInfo rp_begin{};
    rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin.renderPass = render_pass;
    rp_begin.framebuffer = framebuffer;
    rp_begin.renderArea.offset = { 0, 0 };
    rp_begin.renderArea.extent = extent;
    rp_begin.clearValueCount = 1;
    rp_begin.pClearValues = &clear;

    vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdDraw(cmd, vertex_count, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) { throw std::runtime_error("vkEndCommandBuffer failed"); }
  }

private:
  VkShaderModule create_module(const std::vector<std::uint32_t> &spirv) const
  {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spirv.size() * sizeof(std::uint32_t);
    ci.pCode = spirv.data();
    VkShaderModule module{ VK_NULL_HANDLE };
    if (vkCreateShaderModule(device_, &ci, nullptr, &module) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateShaderModule failed");
    }
    return module;
  }

  VkDevice device_{ VK_NULL_HANDLE };
  VkPipelineLayout layout_{ VK_NULL_HANDLE };
  VkPipeline pipeline_{ VK_NULL_HANDLE };
};

} // namespace vkexec
