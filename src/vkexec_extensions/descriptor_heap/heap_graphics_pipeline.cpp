#include <vkexec_extensions/descriptor_heap/heap_graphics_pipeline.hpp>

#include <vkexec/context.hpp>
#include <vkexec/detail/descriptor_backend.hpp>
#include <vkexec/detail/shader_module.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/spirv_compile.hpp>
#include <vkexec_extensions/descriptor_heap/strategy.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vkexec {
namespace {

  auto apply_blend(blend_mode mode, VkPipelineColorBlendAttachmentState &attachment) -> void
  {
    attachment.colorWriteMask = static_cast<VkColorComponentFlags>(
      static_cast<std::uint32_t>(VK_COLOR_COMPONENT_R_BIT) | static_cast<std::uint32_t>(VK_COLOR_COMPONENT_G_BIT)
      | static_cast<std::uint32_t>(VK_COLOR_COMPONENT_B_BIT) | static_cast<std::uint32_t>(VK_COLOR_COMPONENT_A_BIT));
    switch (mode) {
    case blend_mode::none:
      attachment.blendEnable = VK_FALSE;
      break;
    case blend_mode::alpha:
      attachment.blendEnable = VK_TRUE;
      attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      attachment.colorBlendOp = VK_BLEND_OP_ADD;
      attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      attachment.alphaBlendOp = VK_BLEND_OP_ADD;
      break;
    case blend_mode::premultiplied:
      attachment.blendEnable = VK_TRUE;
      attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
      attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      attachment.colorBlendOp = VK_BLEND_OP_ADD;
      attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      attachment.alphaBlendOp = VK_BLEND_OP_ADD;
      break;
    }
  }

  // NOLINTBEGIN(bugprone-easily-swappable-parameters)
  auto create_heap_graphics_vk_pipeline(VkDevice device,
    VkShaderModule vert_module,
    VkShaderModule frag_module,
    VkPipelineLayout layout,
    heap_graphics_layout_desc const &desc) -> result<VkPipeline>
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

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = desc.topology;

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
    depth_stencil.depthTestEnable = desc.depth_test ? VK_TRUE : VK_FALSE;
    depth_stencil.depthWriteEnable = (desc.depth_test && desc.depth_write) ? VK_TRUE : VK_FALSE;
    depth_stencil.depthCompareOp = desc.depth_test ? VK_COMPARE_OP_LESS : VK_COMPARE_OP_ALWAYS;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;

    std::vector<VkPipelineColorBlendAttachmentState> blend_attachments(desc.color_formats.size());
    for (VkPipelineColorBlendAttachmentState &attachment : blend_attachments) { apply_blend(desc.blend, attachment); }

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = static_cast<std::uint32_t>(blend_attachments.size());
    blend.pAttachments = blend_attachments.data();

    std::array dynamic_states{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();

    VkPipelineRenderingCreateInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.colorAttachmentCount = static_cast<std::uint32_t>(desc.color_formats.size());
    rendering.pColorAttachmentFormats = desc.color_formats.data();
    rendering.depthAttachmentFormat = desc.depth_format;
    rendering.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

    VkPipelineCreateFlags2CreateInfo flags2{};
    flags2.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO;
    flags2.pNext = &rendering;
    flags2.flags = detail::heap_descriptor_backend::pipeline_create_flags();

    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.pNext = &flags2;
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
    gpci.renderPass = VK_NULL_HANDLE;
    gpci.subpass = 0;

    VkPipeline pipeline{ VK_NULL_HANDLE };
    if (VkResult const created = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline);
      created != VK_SUCCESS) {
      return fail(created, "vkCreateGraphicsPipelines failed (heap)");
    }
    return pipeline;
  }

}// namespace

auto destroy_heap_graphics_resources(context const &ctx, pipeline_resources &resources) noexcept -> void
{
  VkDevice device = ctx.device();
  if (resources.pipeline != VK_NULL_HANDLE) { vkDestroyPipeline(device, resources.pipeline, nullptr); }
  detail::heap_descriptor_backend::destroy_binding_objects(device, resources);
  if (resources.shader != VK_NULL_HANDLE) { vkDestroyShaderModule(device, resources.shader, nullptr); }
  resources = {};
}

auto create_heap_graphics_resources(context &ctx,
  std::span<std::uint32_t const> vertex_spirv,
  std::span<std::uint32_t const> fragment_spirv,
  heap_graphics_layout_desc const &desc) -> result<pipeline_resources>
{
  if (vertex_spirv.empty() || fragment_spirv.empty()) {
    return fail(errc::invalid_argument, "create_heap_graphics_resources requires non-empty SPIR-V");
  }
  if (desc.color_formats.empty()) {
    return fail(errc::invalid_argument, "create_heap_graphics_resources requires at least one color format");
  }

  pipeline_resources resources{};
  VkDevice device = ctx.device();
  detail::descriptor_layout_info const layout_info{ .bindings = {},
    .push_stages = 0,
    .push_bytes = 0,
    .sets_per_pool = detail::k_descriptor_sets_per_pool,
    .create_set_layout = false,
    .create_pool = false };
  if (auto created = detail::heap_descriptor_backend::create_set_and_pipeline_layout(device, resources, layout_info);
    !created) {
    return fail(created);
  }
  if (auto created = detail::heap_descriptor_backend::create_descriptor_pool(device, resources, layout_info);
    !created) {
    destroy_heap_graphics_resources(ctx, resources);
    return fail(created);
  }

  auto vert_result = detail::create_shader_module(device, vertex_spirv);
  if (!vert_result) { return fail(vert_result); }
  VkShaderModule vert_module = expected_take(vert_result);

  auto frag_result = detail::create_shader_module(device, fragment_spirv);
  if (!frag_result) {
    vkDestroyShaderModule(device, vert_module, nullptr);
    return fail(frag_result);
  }
  VkShaderModule frag_module = expected_take(frag_result);

  auto pipeline_result =
    create_heap_graphics_vk_pipeline(device, vert_module, frag_module, resources.pipeline_layout, desc);
  vkDestroyShaderModule(device, frag_module, nullptr);
  vkDestroyShaderModule(device, vert_module, nullptr);
  if (!pipeline_result) { return fail(pipeline_result); }
  resources.pipeline = expected_take(pipeline_result);
  return resources;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
auto create_heap_graphics_resources(context &ctx,
  std::string_view vertex_glsl,
  std::string_view fragment_glsl,
  heap_graphics_layout_desc const &desc,
  std::string_view vertex_name,
  std::string_view fragment_name) -> result<pipeline_resources>
// NOLINTEND(bugprone-easily-swappable-parameters)
{
  if (vertex_glsl.empty() || fragment_glsl.empty()) {
    return fail(errc::invalid_argument, "create_heap_graphics_resources requires non-empty GLSL");
  }
  VKEXEC_TRY_ASSIGN(
    vert_spirv, compile_glsl_to_spirv(vertex_glsl, vertex_name, shader_kind::vertex, ctx.api_version()));
  VKEXEC_TRY_ASSIGN(
    frag_spirv, compile_glsl_to_spirv(fragment_glsl, fragment_name, shader_kind::fragment, ctx.api_version()));
  return create_heap_graphics_resources(ctx, vert_spirv, frag_spirv, desc);
}

auto heap_graphics_pipeline::reset() noexcept -> void
{
  if (ctx_ != nullptr && resources_ != nullptr) { destroy_heap_graphics_resources(*ctx_, *resources_); }
  resources_.reset();
  ctx_ = nullptr;
}

auto heap_graphics_pipeline::create(context &ctx,
  std::span<std::uint32_t const> vertex_spirv,
  std::span<std::uint32_t const> fragment_spirv,
  heap_graphics_layout_desc const &desc) -> detail::sync_sender_fn<heap_graphics_pipeline>
{
  return detail::make_sync_sender_fn<heap_graphics_pipeline>(
    [&ctx, vertex_spirv, fragment_spirv, desc]() -> result<heap_graphics_pipeline> {
      VKEXEC_TRY_ASSIGN(owned, create_heap_graphics_resources(ctx, vertex_spirv, fragment_spirv, desc));
      return make(ctx, std::make_unique<pipeline_resources>(owned));
    });
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
auto heap_graphics_pipeline::create(context &ctx,
  std::string_view vertex_glsl,
  std::string_view fragment_glsl,
  heap_graphics_layout_desc const &desc,
  std::string_view vertex_name,
  std::string_view fragment_name) -> detail::sync_sender_fn<heap_graphics_pipeline>
// NOLINTEND(bugprone-easily-swappable-parameters)
{
  return detail::make_sync_sender_fn<heap_graphics_pipeline>(
    [&ctx,
      vertex_glsl = std::string(vertex_glsl),
      fragment_glsl = std::string(fragment_glsl),
      desc,
      vertex_name = std::string(vertex_name),
      fragment_name = std::string(fragment_name)]() -> result<heap_graphics_pipeline> {
      VKEXEC_TRY_ASSIGN(
        owned, create_heap_graphics_resources(ctx, vertex_glsl, fragment_glsl, desc, vertex_name, fragment_name));
      return make(ctx, std::make_unique<pipeline_resources>(owned));
    });
}

}// namespace vkexec
