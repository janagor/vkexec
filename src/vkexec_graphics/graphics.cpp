#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>
#include <vkexec_graphics/mesh.hpp>

#include <vkexec/context.hpp>
#include <vkexec/detail/descriptor_backend.hpp>
#include <vkexec/detail/shader_module.hpp>
#include <vkexec/detail/record_with_binding.hpp>
#include <vkexec/detail/viewport.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/spirv_compile.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
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

  constexpr std::uint32_t k_graphics_descriptor_sets_per_pool = 64;

  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  auto build_graphics_pipeline(VkDevice device,
    VkRenderPass render_pass,
    graphics_pipeline_config const &cfg,
    VkPipelineLayout layout,
    VkShaderModule vert_module,
    VkShaderModule frag_module) -> result<VkPipeline>
  // NOLINTEND(bugprone-easily-swappable-parameters)
  {
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
    if (cfg.use_mesh_vertices) {
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
    input_assembly.topology = cfg.topology;

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
    depth_stencil.depthTestEnable = cfg.depth_test ? VK_TRUE : VK_FALSE;
    depth_stencil.depthWriteEnable = (cfg.depth_test && cfg.depth_write) ? VK_TRUE : VK_FALSE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = static_cast<VkColorComponentFlags>(
      static_cast<std::uint32_t>(VK_COLOR_COMPONENT_R_BIT) | static_cast<std::uint32_t>(VK_COLOR_COMPONENT_G_BIT)
      | static_cast<std::uint32_t>(VK_COLOR_COMPONENT_B_BIT) | static_cast<std::uint32_t>(VK_COLOR_COMPONENT_A_BIT));
    if (cfg.alpha_blend) {
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
    gpci.layout = layout;
    gpci.renderPass = render_pass;
    gpci.subpass = 0;
    VkPipeline pipeline{ VK_NULL_HANDLE };
    if (VkResult const created = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline);
      created != VK_SUCCESS) {
      return fail(created, "vkCreateGraphicsPipelines failed");
    }
    return pipeline;
  }

  auto install_storage_set(context &ctx,
    graphics_pipeline_resources const &resources,
    std::span<storage_binding const> buffers) -> result<VkDescriptorSet>
  {
    VKEXEC_TRY_ASSIGN(bound, bind_graphics_storage(ctx, resources, buffers));
    return bound.set;
  }

  auto make_graphics_pipeline(context &ctx,
    VkRenderPass render_pass,
    graphics_pipeline_config cfg,
    std::span<std::uint32_t const> vs_spv,
    std::span<std::uint32_t const> fs_spv,
    std::span<storage_binding const> buffers) -> result<graphics_pipeline>
  {
    auto const binding_count = static_cast<std::uint32_t>(buffers.size());
    VKEXEC_TRY_ASSIGN(owned, create_graphics_resources(ctx, render_pass, cfg, vs_spv, fs_spv, binding_count));
    VKEXEC_TRY_ASSIGN(set, install_storage_set(ctx, owned, buffers));
    return graphics_pipeline::make(
      ctx, std::make_unique<graphics_pipeline_resources>(owned), set, std::vector(buffers.begin(), buffers.end()));
  }

}// namespace

auto destroy_graphics_resources(context const &ctx, graphics_pipeline_resources &resources) noexcept -> void
{
  VkDevice device = ctx.device();
  if (resources.pipeline != VK_NULL_HANDLE) { vkDestroyPipeline(device, resources.pipeline, nullptr); }
  if (resources.pipeline_layout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device, resources.pipeline_layout, nullptr);
  }
  if (resources.descriptor_pool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device, resources.descriptor_pool, nullptr);
  }
  if (resources.set_layout != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device, resources.set_layout, nullptr); }
  resources = {};
}

auto create_graphics_resources(context &ctx,
  VkRenderPass render_pass,
  graphics_pipeline_config cfg,
  std::span<std::uint32_t const> vertex_spirv,
  std::span<std::uint32_t const> fragment_spirv,
  std::uint32_t storage_binding_count) -> result<graphics_pipeline_resources>
{
  if (vertex_spirv.empty() || fragment_spirv.empty()) {
    return fail(errc::invalid_argument, "create_graphics_resources requires non-empty SPIR-V");
  }

  graphics_pipeline_resources owned{};
  owned.cfg = cfg;
  owned.binding_count = storage_binding_count;
  VkDevice device = ctx.device();

  if (storage_binding_count > 0) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(storage_binding_count);
    for (std::uint32_t index = 0; index < storage_binding_count; ++index) {
      bindings.at(index).binding = index;
      bindings.at(index).descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings.at(index).descriptorCount = 1;
      bindings.at(index).stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = storage_binding_count;
    dslci.pBindings = bindings.data();
    if (VkResult const result = vkCreateDescriptorSetLayout(device, &dslci, nullptr, &owned.set_layout);
      result != VK_SUCCESS) {
      return fail(result, "vkCreateDescriptorSetLayout failed (graphics)");
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = storage_binding_count * k_graphics_descriptor_sets_per_pool;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = k_graphics_descriptor_sets_per_pool;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &pool_size;
    if (VkResult const result = vkCreateDescriptorPool(device, &dpci, nullptr, &owned.descriptor_pool);
      result != VK_SUCCESS) {
      destroy_graphics_resources(ctx, owned);
      return fail(result, "vkCreateDescriptorPool failed (graphics)");
    }
  }

  auto vert = detail::create_shader_module(device, vertex_spirv);
  if (!vert) {
    destroy_graphics_resources(ctx, owned);
    return fail(vert);
  }
  VkShaderModule vert_module = expected_take(vert);
  auto frag = detail::create_shader_module(device, fragment_spirv);
  if (!frag) {
    vkDestroyShaderModule(device, vert_module, nullptr);
    destroy_graphics_resources(ctx, owned);
    return fail(frag);
  }
  VkShaderModule frag_module = expected_take(frag);

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  if (owned.set_layout != VK_NULL_HANDLE) {
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &owned.set_layout;
  }
  if (VkResult const layout_result = vkCreatePipelineLayout(device, &plci, nullptr, &owned.pipeline_layout);
    layout_result != VK_SUCCESS) {
    vkDestroyShaderModule(device, frag_module, nullptr);
    vkDestroyShaderModule(device, vert_module, nullptr);
    destroy_graphics_resources(ctx, owned);
    return fail(layout_result, "vkCreatePipelineLayout failed");
  }

  auto pipeline = build_graphics_pipeline(device, render_pass, cfg, owned.pipeline_layout, vert_module, frag_module);
  vkDestroyShaderModule(device, frag_module, nullptr);
  vkDestroyShaderModule(device, vert_module, nullptr);
  if (!pipeline) {
    destroy_graphics_resources(ctx, owned);
    return fail(pipeline);
  }
  owned.pipeline = expected_take(pipeline);
  return owned;
}

auto create_graphics_resources(context &ctx,
  VkRenderPass render_pass,
  graphics_pipeline_config cfg,
  std::string_view vertex_glsl,
  std::string_view fragment_glsl,
  std::uint32_t storage_binding_count) -> result<graphics_pipeline_resources>
{
  if (vertex_glsl.empty() || fragment_glsl.empty()) {
    return fail(errc::invalid_argument, "create_graphics_resources requires non-empty GLSL");
  }
  VKEXEC_TRY_ASSIGN(vs_spv, compile_glsl_to_spirv(vertex_glsl, "vkexec.vert", shader_kind::vertex, ctx.api_version()));
  VKEXEC_TRY_ASSIGN(
    fs_spv, compile_glsl_to_spirv(fragment_glsl, "vkexec.frag", shader_kind::fragment, ctx.api_version()));
  return create_graphics_resources(ctx, render_pass, cfg, vs_spv, fs_spv, storage_binding_count);
}

auto allocate_graphics_set(context const &ctx, graphics_pipeline_resources const &pipe) -> result<VkDescriptorSet>
{
  if (pipe.descriptor_pool == VK_NULL_HANDLE || pipe.set_layout == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "allocate_graphics_set requires a descriptor pool and set layout");
  }
  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = pipe.descriptor_pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &pipe.set_layout;
  VkDescriptorSet set{ VK_NULL_HANDLE };
  if (VkResult const result = vkAllocateDescriptorSets(ctx.device(), &dsai, &set); result != VK_SUCCESS) {
    return fail(result, "vkAllocateDescriptorSets failed (graphics)");
  }
  return set;
}

auto bind_graphics_storage(context &ctx,
  graphics_pipeline_resources const &pipe,
  std::span<storage_binding const> buffers) -> result<bound_graphics>
{
  if (buffers.size() != pipe.binding_count) {
    return fail(errc::invalid_argument, "bind_graphics_storage buffer count must match binding_count");
  }
  if (buffers.empty()) { return bound_graphics{ .pipe = &pipe, .set = VK_NULL_HANDLE }; }
  VKEXEC_TRY_ASSIGN(set, allocate_graphics_set(ctx, pipe));
  write_storage_descriptors(ctx.device(), set, buffers);
  return bound_graphics{ .pipe = &pipe, .set = set };
}

auto free_graphics_set(context const &ctx, graphics_pipeline_resources const &pipe, VkDescriptorSet set) noexcept
  -> void
{
  if (set == VK_NULL_HANDLE || pipe.descriptor_pool == VK_NULL_HANDLE) { return; }
  vkFreeDescriptorSets(ctx.device(), pipe.descriptor_pool, 1, &set);
}

auto begin_graphics_pass(VkCommandBuffer cmd,
  VkRenderPass render_pass,
  VkFramebuffer framebuffer,
  VkExtent2D extent,
  graphics_pipeline_config const &cfg) -> void
{
  std::array<VkClearValue, k_graphics_clear_count> const clears = make_clear_values(cfg);
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

namespace {

  auto bind_graphics_draw_state(VkCommandBuffer cmd, graphics_bind bind, VkExtent2D extent) -> void
  {
    (void)detail::bind_and_push<detail::set_descriptor_backend>(
      nullptr, cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bind, {});

    detail::set_dynamic_viewport_scissor(cmd, extent);
  }

}// namespace

auto record_draw(VkCommandBuffer cmd, graphics_bind bind, VkExtent2D extent, std::uint32_t vertex_count) -> void
{
  bind_graphics_draw_state(cmd, bind, extent);
  vkCmdDraw(cmd, vertex_count, 1, 0, 0);
}

auto record_draw(VkCommandBuffer cmd, graphics_bind bind, VkExtent2D extent, mesh_draw const &drawn) -> void
{
  bind_graphics_draw_state(cmd, bind, extent);
  VkDeviceSize const vertex_offset = 0;
  vkCmdBindVertexBuffers(cmd, 0, 1, &drawn.vertex_buffer, &vertex_offset);
  vkCmdBindIndexBuffer(cmd, drawn.index_buffer, 0, VK_INDEX_TYPE_UINT32);
  vkCmdDrawIndexed(cmd, drawn.index_count, 1, 0, 0, 0);
}

auto draw_pass(VkCommandBuffer cmd,
  VkRenderPass render_pass,
  VkFramebuffer framebuffer,
  VkExtent2D extent,
  graphics_pipeline_config const &cfg,
  graphics_bind bind,
  std::uint32_t vertex_count) -> status
{
  begin_graphics_pass(cmd, render_pass, framebuffer, extent, cfg);
  record_draw(cmd, bind, extent, vertex_count);
  end_graphics_pass(cmd);
  if (VkResult const result = vkEndCommandBuffer(cmd); result != VK_SUCCESS) {
    return fail(result, "vkEndCommandBuffer failed");
  }
  return {};
}

auto draw_pass(VkCommandBuffer cmd,
  VkRenderPass render_pass,
  VkFramebuffer framebuffer,
  VkExtent2D extent,
  graphics_pipeline_config const &cfg,
  graphics_bind bind,
  mesh_draw const &drawn) -> status
{
  begin_graphics_pass(cmd, render_pass, framebuffer, extent, cfg);
  record_draw(cmd, bind, extent, drawn);
  end_graphics_pass(cmd);
  if (VkResult const result = vkEndCommandBuffer(cmd); result != VK_SUCCESS) {
    return fail(result, "vkEndCommandBuffer failed");
  }
  return {};
}

auto graphics_pipeline::reset() noexcept -> void
{
  if (ctx_ != nullptr && resources_ != nullptr) { destroy_graphics_resources(*ctx_, *resources_); }
  resources_.reset();
  descriptor_set_ = VK_NULL_HANDLE;
  buffers_.clear();
  ctx_ = nullptr;
}

auto graphics_pipeline::create(context &ctx,
  VkRenderPass render_pass,
  graphics_pipeline_config cfg,
  std::span<std::uint32_t const> vertex_spirv,
  std::span<std::uint32_t const> fragment_spirv,
  std::span<storage_binding const> buffers) -> detail::sync_sender_fn<graphics_pipeline>
{
  return detail::make_sync_sender_fn<graphics_pipeline>(
    [&ctx,
      render_pass,
      cfg,
      vertex_spirv = std::vector(vertex_spirv.begin(), vertex_spirv.end()),
      fragment_spirv = std::vector(fragment_spirv.begin(), fragment_spirv.end()),
      owned = std::vector(buffers.begin(), buffers.end())]() mutable -> result<graphics_pipeline> {
      return make_graphics_pipeline(ctx, render_pass, cfg, vertex_spirv, fragment_spirv, owned);
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
  return detail::make_sync_sender_fn<graphics_pipeline>(
    [&ctx,
      render_pass,
      cfg,
      vertex_glsl = std::string(vertex_glsl),
      fragment_glsl = std::string(fragment_glsl),
      owned = std::vector(buffers.begin(), buffers.end())]() mutable -> result<graphics_pipeline> {
      VKEXEC_TRY_ASSIGN(owned_resources,
        create_graphics_resources(
          ctx, render_pass, cfg, vertex_glsl, fragment_glsl, static_cast<std::uint32_t>(owned.size())));
      VKEXEC_TRY_ASSIGN(set, install_storage_set(ctx, owned_resources, owned));
      return graphics_pipeline::make(
        ctx, std::make_unique<graphics_pipeline_resources>(owned_resources), set, std::move(owned));
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

auto graphics_pipeline::record_draw(VkCommandBuffer cmd, VkExtent2D extent, std::uint32_t vertex_count) const -> void
{ vkexec::record_draw(cmd, bind(), extent, vertex_count); }

auto graphics_pipeline::record_draw(VkCommandBuffer cmd, VkExtent2D extent, mesh const &drawn) const -> void
{
  mesh_draw const handles{ .vertex_buffer = drawn.vk_vertex_buffer(),
    .index_buffer = drawn.vk_index_buffer(),
    .index_count = drawn.index_count() };
  vkexec::record_draw(cmd, bind(), extent, handles);
}

auto graphics_pipeline::draw(VkCommandBuffer cmd,
  VkRenderPass render_pass,
  VkFramebuffer framebuffer,
  VkExtent2D extent,
  std::uint32_t vertex_count) const -> status
{ return draw_pass(cmd, render_pass, framebuffer, extent, resources_->cfg, bind(), vertex_count); }

auto graphics_pipeline::draw(VkCommandBuffer cmd,
  VkRenderPass render_pass,
  VkFramebuffer framebuffer,
  VkExtent2D extent,
  mesh const &drawn) const -> status
{
  mesh_draw const handles{ .vertex_buffer = drawn.vk_vertex_buffer(),
    .index_buffer = drawn.vk_index_buffer(),
    .index_count = drawn.index_count() };
  return draw_pass(cmd, render_pass, framebuffer, extent, resources_->cfg, bind(), handles);
}

}// namespace vkexec
