#ifndef VKEXEC_GRAPHICS_HPP
#define VKEXEC_GRAPHICS_HPP


#include <vkexec/context.hpp>
#include <vkexec_edsl/glsl_emit.hpp>
#include <vkexec_edsl/spirv.hpp>
#include <vkexec_edsl/types.hpp>

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vkexec {

constexpr float k_default_clear_r = 0.08F;
constexpr float k_default_clear_g = 0.09F;
constexpr float k_default_clear_b = 0.12F;
constexpr float k_default_clear_a = 1.0F;

struct graphics_pipeline_config
{
  VkPrimitiveTopology topology{ VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
  bool alpha_blend{ false };
  float clear_r{ k_default_clear_r };
  float clear_g{ k_default_clear_g };
  float clear_b{ k_default_clear_b };
  float clear_a{ k_default_clear_a };
};

/// Graphics pipeline built by tracing vertex/fragment eDSL lambdas to GLSL/SPIR-V.
class graphics_pipeline
{
public:
  template<typename VertexFn, typename FragmentFn>
  graphics_pipeline(context &ctx,
    VkRenderPass render_pass,
    graphics_pipeline_config cfg,
    VertexFn &&vertex_fn,
    FragmentFn &&fragment_fn)
    : device_(ctx.device()), cfg_(cfg)
  { build(ctx, render_pass, std::forward<VertexFn>(vertex_fn), std::forward<FragmentFn>(fragment_fn)); }

  template<typename VertexFn, typename FragmentFn>
  graphics_pipeline(context &ctx, VkRenderPass render_pass, VertexFn &&vertex_fn, FragmentFn &&fragment_fn)
    : graphics_pipeline(ctx,
        render_pass,
        graphics_pipeline_config{},
        std::forward<VertexFn>(vertex_fn),
        std::forward<FragmentFn>(fragment_fn))
  {}

  ~graphics_pipeline() { destroy(); }

  graphics_pipeline(const graphics_pipeline &) = delete;
  auto operator=(const graphics_pipeline &) -> graphics_pipeline & = delete;

  graphics_pipeline(graphics_pipeline &&other) noexcept
    : device_(other.device_), cfg_(other.cfg_), layout_(other.layout_), pipeline_(other.pipeline_),
      set_layout_(other.set_layout_), descriptor_pool_(other.descriptor_pool_), descriptor_set_(other.descriptor_set_),
      buffers_(std::move(other.buffers_))
  { other.release(); }

  auto operator=(graphics_pipeline &&other) noexcept -> graphics_pipeline &
  {
    if (this == &other) { return *this; }
    destroy();
    device_ = other.device_;
    cfg_ = other.cfg_;
    layout_ = other.layout_;
    pipeline_ = other.pipeline_;
    set_layout_ = other.set_layout_;
    descriptor_pool_ = other.descriptor_pool_;
    descriptor_set_ = other.descriptor_set_;
    buffers_ = std::move(other.buffers_);
    other.release();
    return *this;
  }

  [[nodiscard]] auto pipeline() const noexcept -> VkPipeline { return pipeline_; }
  [[nodiscard]] auto config() const noexcept -> graphics_pipeline_config const & { return cfg_; }

  /// Bind pipeline, descriptors, viewport/scissor, and issue `vkCmdDraw` (no render-pass management).
  auto record_draw(VkCommandBuffer cmd, VkExtent2D extent, std::uint32_t vertex_count) const -> void
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

    if (descriptor_set_ != VK_NULL_HANDLE) {
      std::vector<VkDescriptorBufferInfo> infos(buffers_.size());
      std::vector<VkWriteDescriptorSet> writes(buffers_.size());
      for (std::size_t index = 0; index < buffers_.size(); ++index) {
        infos.at(index).buffer = buffers_.at(index).buffer;
        infos.at(index).offset = 0;
        infos.at(index).range = buffers_.at(index).byte_size;
        writes.at(index).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes.at(index).dstSet = descriptor_set_;
        writes.at(index).dstBinding = buffers_.at(index).binding;
        writes.at(index).descriptorCount = 1;
        writes.at(index).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes.at(index).pBufferInfo = &infos.at(index);
      }
      vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &descriptor_set_, 0, nullptr);
    }

    VkViewport viewport{};
    viewport.x = 0.0F;
    viewport.y = 0.0F;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0F;
    viewport.maxDepth = 1.0F;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { .x = 0, .y = 0 };
    scissor.extent = extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdDraw(cmd, vertex_count, 1, 0, 0);
  }

  /// Begin the render pass, bind this pipeline, draw `vertex_count` verts, end the pass, and end the cmd buffer.
  auto draw(VkCommandBuffer cmd,
    VkRenderPass render_pass,
    VkFramebuffer framebuffer,
    VkExtent2D extent,
    std::uint32_t vertex_count) const -> void
  {
    VkClearValue clear{};
    clear.color = { { cfg_.clear_r, cfg_.clear_g, cfg_.clear_b, cfg_.clear_a } };

    VkRenderPassBeginInfo rp_begin{};
    rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin.renderPass = render_pass;
    rp_begin.framebuffer = framebuffer;
    rp_begin.renderArea.offset = { .x = 0, .y = 0 };
    rp_begin.renderArea.extent = extent;
    rp_begin.clearValueCount = 1;
    rp_begin.pClearValues = &clear;

    vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
    record_draw(cmd, extent, vertex_count);
    vkCmdEndRenderPass(cmd);
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) { throw std::runtime_error("vkEndCommandBuffer failed"); }
  }

private:
  struct bound_buffer
  {
    std::uint32_t binding{ 0 };
    VkBuffer buffer{ VK_NULL_HANDLE };
    VkDeviceSize byte_size{ 0 };
  };

  auto destroy() noexcept -> void
  {
    if (device_ == VK_NULL_HANDLE) { return; }
    if (pipeline_ != VK_NULL_HANDLE) { vkDestroyPipeline(device_, pipeline_, nullptr); }
    if (layout_ != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device_, layout_, nullptr); }
    if (set_layout_ != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr); }
    if (descriptor_pool_ != VK_NULL_HANDLE) { vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr); }
  }

  auto release() noexcept -> void
  {
    device_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE;
    descriptor_pool_ = VK_NULL_HANDLE;
    descriptor_set_ = VK_NULL_HANDLE;
  }

  template<typename VertexFn, typename FragmentFn>
  auto build(context &ctx, VkRenderPass render_pass, VertexFn &&vertex_fn, FragmentFn &&fragment_fn) -> void
  {
    edsl::ASTContext vs_ast;
    {
      const edsl::ASTScope scope(vs_ast);
      const edsl::Int vertex_id = edsl::Int::vertex_index();
      const edsl::VertexWriter vertex_out;
      std::forward<VertexFn>(vertex_fn)(vertex_id, vertex_out);
    }
    const std::string vs_glsl = edsl::emit_vertex_glsl(vs_ast);
    const auto vs_spv = edsl::compile_glsl_to_spirv(vs_glsl, "vkexec.vert", edsl::shader_kind::vertex);

    edsl::ASTContext fs_ast;
    {
      const edsl::ASTScope scope(fs_ast);
      const edsl::FragmentReader fragment_in;
      const edsl::FragmentWriter fragment_out;
      std::forward<FragmentFn>(fragment_fn)(fragment_in, fragment_out);
    }
    const std::string fs_glsl = edsl::emit_fragment_glsl(fs_ast);
    const auto fs_spv = edsl::compile_glsl_to_spirv(fs_glsl, "vkexec.frag", edsl::shader_kind::fragment);

    std::ranges::transform(
      vs_ast.buffers, std::back_inserter(buffers_), [](const edsl::BufferBinding &buffer_binding) -> bound_buffer {
        return bound_buffer{
          .binding = static_cast<std::uint32_t>(buffer_binding.binding),
          .buffer = static_cast<VkBuffer>(buffer_binding.vk_buffer),
          .byte_size = static_cast<VkDeviceSize>(buffer_binding.byte_size),
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
    const VkResult created = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline_);
    vkDestroyShaderModule(device_, frag, nullptr);
    vkDestroyShaderModule(device_, vert, nullptr);
    if (created != VK_SUCCESS) { throw std::runtime_error("vkCreateGraphicsPipelines failed"); }
    (void)ctx;
  }

  [[nodiscard]] auto create_module(const std::vector<std::uint32_t> &spirv) const -> VkShaderModule
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

  VkDevice device_{ VK_NULL_HANDLE };
  graphics_pipeline_config cfg_{};
  VkPipelineLayout layout_{ VK_NULL_HANDLE };
  VkPipeline pipeline_{ VK_NULL_HANDLE };
  VkDescriptorSetLayout set_layout_{ VK_NULL_HANDLE };
  VkDescriptorPool descriptor_pool_{ VK_NULL_HANDLE };
  VkDescriptorSet descriptor_set_{ VK_NULL_HANDLE };
  std::vector<bound_buffer> buffers_;
};

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_HPP
