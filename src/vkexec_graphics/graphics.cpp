#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/mesh.hpp>

#include <vkexec/context.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/spirv_compile.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <span>
#include <string_view>
#include <vector>

namespace vkexec {

auto graphics_pipeline::destroy() noexcept -> void
{
  if (device_ == VK_NULL_HANDLE) { return; }
  if (pipeline_ != VK_NULL_HANDLE) { vkDestroyPipeline(device_, pipeline_, nullptr); }
  if (layout_ != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device_, layout_, nullptr); }
  if (set_layout_ != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr); }
  if (descriptor_pool_ != VK_NULL_HANDLE) { vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr); }
  release();
}

auto graphics_pipeline::release() noexcept -> void
{
  device_ = VK_NULL_HANDLE;
  layout_ = VK_NULL_HANDLE;
  pipeline_ = VK_NULL_HANDLE;
  set_layout_ = VK_NULL_HANDLE;
  descriptor_pool_ = VK_NULL_HANDLE;
  descriptor_set_ = VK_NULL_HANDLE;
}

auto graphics_pipeline::create(context &ctx,
  VkRenderPass render_pass,
  graphics_pipeline_config cfg,
  std::span<std::uint32_t const> vertex_spirv,
  std::span<std::uint32_t const> fragment_spirv,
  std::span<storage_binding const> buffers) -> detail::sync_sender_fn<graphics_pipeline>
{
  auto const make = [](context &host,
                      VkRenderPass pass,
                      graphics_pipeline_config pipe_cfg,
                      std::vector<std::uint32_t> const &vs_spv,
                      std::vector<std::uint32_t> const &fs_spv,
                      std::span<storage_binding const> pipe_buffers) -> result<graphics_pipeline> {
    if (vs_spv.empty() || fs_spv.empty()) {
      return fail(errc::invalid_argument, "graphics_pipeline::create requires non-empty SPIR-V");
    }
    graphics_pipeline pipe;
    pipe.device_ = host.device();
    pipe.cfg_ = pipe_cfg;
    if (auto built = pipe.complete(host, pass, vs_spv, fs_spv, pipe_buffers); !built) {
      pipe.destroy();
      return fail(built);
    }
    return pipe;
  };

  return detail::make_sync_sender_fn<graphics_pipeline>(
    [&ctx,
      render_pass,
      cfg,
      // Own SPIR-V/bindings: the sender may run after the caller's spans are gone.
      vertex_spirv = std::vector(vertex_spirv.begin(), vertex_spirv.end()),
      fragment_spirv = std::vector(fragment_spirv.begin(), fragment_spirv.end()),
      owned = std::vector(buffers.begin(), buffers.end()),
      make]() mutable -> result<graphics_pipeline> {
      return make(ctx, render_pass, cfg, vertex_spirv, fragment_spirv, owned);
    });
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
auto graphics_pipeline::create(context &ctx,
  VkRenderPass render_pass,
  graphics_pipeline_config cfg,
  std::string_view vertex_glsl,
  std::string_view fragment_glsl,
  std::span<storage_binding const> buffers) -> detail::sync_sender_fn<graphics_pipeline>
// NOLINTEND(bugprone-easily-swappable-parameters)
{
  auto const make = [](context &host,
                      VkRenderPass pass,
                      graphics_pipeline_config pipe_cfg,
                      std::vector<std::uint32_t> const &vs_spv,
                      std::vector<std::uint32_t> const &fs_spv,
                      std::span<storage_binding const> pipe_buffers) -> result<graphics_pipeline> {
    if (vs_spv.empty() || fs_spv.empty()) {
      return fail(errc::invalid_argument, "graphics_pipeline::create requires non-empty SPIR-V");
    }
    graphics_pipeline pipe;
    pipe.device_ = host.device();
    pipe.cfg_ = pipe_cfg;
    if (auto built = pipe.complete(host, pass, vs_spv, fs_spv, pipe_buffers); !built) {
      pipe.destroy();
      return fail(built);
    }
    return pipe;
  };

  return detail::make_sync_sender_fn<graphics_pipeline>([&ctx,
                                                          render_pass,
                                                          cfg,
                                                          vertex_glsl = std::string(vertex_glsl),
                                                          fragment_glsl = std::string(fragment_glsl),
                                                          owned = std::vector(buffers.begin(), buffers.end()),
                                                          make]() mutable -> result<graphics_pipeline> {
    if (vertex_glsl.empty() || fragment_glsl.empty()) {
      return fail(errc::invalid_argument, "graphics_pipeline::create requires non-empty GLSL");
    }
    VKEXEC_TRY_ASSIGN(
      vs_spv, compile_glsl_to_spirv(vertex_glsl, "vkexec.vert", shader_kind::vertex, ctx.api_version()));
    VKEXEC_TRY_ASSIGN(
      fs_spv, compile_glsl_to_spirv(fragment_glsl, "vkexec.frag", shader_kind::fragment, ctx.api_version()));
    return make(ctx, render_pass, cfg, vs_spv, fs_spv, owned);
  });
}

auto graphics_pipeline::create(context &ctx,
  VkRenderPass render_pass,
  std::span<std::uint32_t const> vertex_spirv,
  std::span<std::uint32_t const> fragment_spirv,
  std::span<storage_binding const> buffers) -> detail::sync_sender_fn<graphics_pipeline>
{ return create(ctx, render_pass, graphics_pipeline_config{}, vertex_spirv, fragment_spirv, buffers); }

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
auto graphics_pipeline::create(context &ctx,
  VkRenderPass render_pass,
  std::string_view vertex_glsl,
  std::string_view fragment_glsl,
  std::span<storage_binding const> buffers) -> detail::sync_sender_fn<graphics_pipeline>
// NOLINTEND(bugprone-easily-swappable-parameters)
{ return create(ctx, render_pass, graphics_pipeline_config{}, vertex_glsl, fragment_glsl, buffers); }

auto graphics_pipeline::create_module(std::vector<std::uint32_t> const &spirv) const -> result<VkShaderModule>
{
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = spirv.size() * sizeof(std::uint32_t);
  create_info.pCode = spirv.data();
  VkShaderModule module{ VK_NULL_HANDLE };
  if (VkResult const result = vkCreateShaderModule(device_, &create_info, nullptr, &module); result != VK_SUCCESS) {
    return fail(result, "vkCreateShaderModule failed");
  }
  return module;
}

auto graphics_pipeline::complete([[maybe_unused]] context &ctx,
  VkRenderPass render_pass,
  std::vector<std::uint32_t> const &vs_spv,
  std::vector<std::uint32_t> const &fs_spv,
  std::span<storage_binding const> buffers) -> status
{
  buffers_.clear();
  buffers_.reserve(buffers.size());
  std::ranges::transform(buffers, std::back_inserter(buffers_), [](storage_binding const &buffer) -> bound_buffer {
    return bound_buffer{
      .binding = buffer.binding,
      .buffer = buffer.buffer,
      .byte_size = buffer.byte_size,
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
    if (VkResult const result = vkCreateDescriptorSetLayout(device_, &dslci, nullptr, &set_layout_);
      result != VK_SUCCESS) {
      return fail(result, "vkCreateDescriptorSetLayout failed (graphics)");
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = static_cast<std::uint32_t>(buffers_.size());
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &pool_size;
    if (VkResult const result = vkCreateDescriptorPool(device_, &dpci, nullptr, &descriptor_pool_);
      result != VK_SUCCESS) {
      return fail(result, "vkCreateDescriptorPool failed (graphics)");
    }

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = descriptor_pool_;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &set_layout_;
    if (VkResult const result = vkAllocateDescriptorSets(device_, &dsai, &descriptor_set_); result != VK_SUCCESS) {
      return fail(result, "vkAllocateDescriptorSets failed (graphics)");
    }
  }

  VKEXEC_TRY_ASSIGN(vert_module, create_module(vs_spv));
  auto frag = create_module(fs_spv);
  if (!frag) {
    vkDestroyShaderModule(device_, vert_module, nullptr);
    return fail(frag);
  }
  VkShaderModule frag_module = expected_take(frag);

  static constexpr std::size_t k_graphics_stage_count = 2;
  // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
  std::array<VkPipelineShaderStageCreateInfo, k_graphics_stage_count> stages{};
  stages.at(0).sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages.at(0).stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages.at(0).module = vert_module;
  stages.at(0).pName = "main";
  stages.at(1).sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages.at(1).stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages.at(1).module = frag_module;
  stages.at(1).pName = "main";

  constexpr std::uint32_t k_mesh_binding = 0;
  constexpr std::uint32_t k_mesh_position_location = 0;
  constexpr std::uint32_t k_mesh_color_location = 1;
  constexpr std::uint32_t k_mesh_attribute_count = 2;

  VkVertexInputBindingDescription mesh_binding{};
  std::array<VkVertexInputAttributeDescription, k_mesh_attribute_count> mesh_attributes{};
  VkPipelineVertexInputStateCreateInfo vertex_input{};
  vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  if (cfg_.use_mesh_vertices) {
    mesh_binding.binding = k_mesh_binding;
    mesh_binding.stride = static_cast<std::uint32_t>(sizeof(mesh_vertex));
    mesh_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    mesh_attributes.at(0).location = k_mesh_position_location;
    mesh_attributes.at(0).binding = k_mesh_binding;
    mesh_attributes.at(0).format = VK_FORMAT_R32G32B32_SFLOAT;
    mesh_attributes.at(0).offset = static_cast<std::uint32_t>(offsetof(mesh_vertex, position));
    mesh_attributes.at(1).location = k_mesh_color_location;
    mesh_attributes.at(1).binding = k_mesh_binding;
    mesh_attributes.at(1).format = VK_FORMAT_R32G32B32_SFLOAT;
    mesh_attributes.at(1).offset = static_cast<std::uint32_t>(offsetof(mesh_vertex, color));
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &mesh_binding;
    vertex_input.vertexAttributeDescriptionCount = k_mesh_attribute_count;
    vertex_input.pVertexAttributeDescriptions = mesh_attributes.data();
  }

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

  VkPipelineDepthStencilStateCreateInfo depth_stencil{};
  depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_stencil.depthTestEnable = cfg_.depth_test ? VK_TRUE : VK_FALSE;
  depth_stencil.depthWriteEnable = (cfg_.depth_test && cfg_.depth_write) ? VK_TRUE : VK_FALSE;
  depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
  depth_stencil.depthBoundsTestEnable = VK_FALSE;
  depth_stencil.stencilTestEnable = VK_FALSE;

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
  if (VkResult const layout_result = vkCreatePipelineLayout(device_, &plci, nullptr, &layout_);
    layout_result != VK_SUCCESS) {
    vkDestroyShaderModule(device_, frag_module, nullptr);
    vkDestroyShaderModule(device_, vert_module, nullptr);
    return fail(layout_result, "vkCreatePipelineLayout failed");
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
  gpci.pDepthStencilState = &depth_stencil;
  gpci.pColorBlendState = &blend;
  gpci.pDynamicState = &dynamic;
  gpci.layout = layout_;
  gpci.renderPass = render_pass;
  gpci.subpass = 0;
  VkResult const created = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline_);
  vkDestroyShaderModule(device_, frag_module, nullptr);
  vkDestroyShaderModule(device_, vert_module, nullptr);
  if (created != VK_SUCCESS) { return fail(created, "vkCreateGraphicsPipelines failed"); }
  return {};
}

auto graphics_pipeline::bind_draw_state(VkCommandBuffer cmd, VkExtent2D extent) const -> void
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
}

auto graphics_pipeline::begin_pass(VkCommandBuffer cmd,
  VkRenderPass render_pass,
  VkFramebuffer framebuffer,
  VkExtent2D extent) const -> void
{
  std::array<VkClearValue, k_graphics_clear_count> const clears = make_clear_values(cfg_);
  VkRenderPassBeginInfo rp_begin{};
  rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rp_begin.renderPass = render_pass;
  rp_begin.framebuffer = framebuffer;
  rp_begin.renderArea.offset = { .x = 0, .y = 0 };
  rp_begin.renderArea.extent = extent;
  rp_begin.clearValueCount = k_graphics_clear_count;
  rp_begin.pClearValues = clears.data();
  vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
}

auto graphics_pipeline::record_draw(VkCommandBuffer cmd, VkExtent2D extent, std::uint32_t vertex_count) const -> void
{
  bind_draw_state(cmd, extent);
  vkCmdDraw(cmd, vertex_count, 1, 0, 0);
}

auto graphics_pipeline::record_draw(VkCommandBuffer cmd, VkExtent2D extent, mesh const &drawn) const -> void
{
  bind_draw_state(cmd, extent);
  VkBuffer vertex_buffer = drawn.vk_vertex_buffer();
  VkDeviceSize const vertex_offset = 0;
  vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &vertex_offset);
  vkCmdBindIndexBuffer(cmd, drawn.vk_index_buffer(), 0, VK_INDEX_TYPE_UINT32);
  vkCmdDrawIndexed(cmd, drawn.index_count(), 1, 0, 0, 0);
}

auto graphics_pipeline::draw(VkCommandBuffer cmd,
  VkRenderPass render_pass,
  VkFramebuffer framebuffer,
  VkExtent2D extent,
  std::uint32_t vertex_count) const -> status
{
  begin_pass(cmd, render_pass, framebuffer, extent);
  record_draw(cmd, extent, vertex_count);
  vkCmdEndRenderPass(cmd);
  if (VkResult const result = vkEndCommandBuffer(cmd); result != VK_SUCCESS) {
    return fail(result, "vkEndCommandBuffer failed");
  }
  return {};
}

auto graphics_pipeline::draw(VkCommandBuffer cmd,
  VkRenderPass render_pass,
  VkFramebuffer framebuffer,
  VkExtent2D extent,
  mesh const &drawn) const -> status
{
  begin_pass(cmd, render_pass, framebuffer, extent);
  record_draw(cmd, extent, drawn);
  vkCmdEndRenderPass(cmd);
  if (VkResult const result = vkEndCommandBuffer(cmd); result != VK_SUCCESS) {
    return fail(result, "vkEndCommandBuffer failed");
  }
  return {};
}

}// namespace vkexec
