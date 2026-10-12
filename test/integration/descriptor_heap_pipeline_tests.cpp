// NOLINTBEGIN(readability-function-cognitive-complexity)

#include "test_helpers.hpp"
#include "vma_test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/bind_resources.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/result.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_extensions/descriptor_heap/algorithm.hpp>
#include <vkexec_extensions/descriptor_heap/buffer.hpp>
#include <vkexec_extensions/descriptor_heap/compute_pipeline.hpp>// NOLINT(misc-include-cleaner)
#include <vkexec_extensions/descriptor_heap/descriptor_heap.hpp>
#include <vkexec_extensions/descriptor_heap/extension.hpp>
#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/heap_graphics_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/resource_table.hpp>
#include <vkexec_extensions/descriptor_heap/strategy.hpp>
#include <vkexec_extensions/extension.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/triangle_shaders.hpp>
#include <vkexec_vma/gpu_buffer.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t k_work_count = 64;
constexpr VkDeviceSize k_storage_bytes = 256;

constexpr std::string_view k_heap_compute_glsl = R"(#version 460
layout(local_size_x = 64) in;
void main() {}
)";

constexpr std::string_view k_heap_spec_glsl = R"(#version 460
layout(local_size_x = 64) in;
layout(constant_id = 0) const uint kCount = 1;
void main() {
  if (gl_GlobalInvocationID.x >= kCount) { return; }
}
)";

constexpr std::string_view k_heap_write_glsl = R"(#version 460
#extension GL_EXT_descriptor_heap : require
layout(local_size_x = 1) in;
layout(descriptor_heap) buffer Storage { uint value[]; } buffers[];
void main() { buffers[0].value[0] += 1u; }
)";

constexpr std::uint32_t k_initial_heap_value = 0;
constexpr std::uint32_t k_expected_heap_value = 2;
constexpr std::uint32_t k_single_workgroup = 1;

struct heap_push
{
  std::uint32_t count;
};

constexpr std::size_t k_fixed_heap_byte_count = 16;
using fixed_heap_byte_span = std::span<std::byte const, k_fixed_heap_byte_count>;

static_assert(vkexec::detail::sender_adaptor_closure<decltype(vkexec::bind_resources(vkexec::descriptor_heap,
    std::declval<vkexec::handles::compute_pipeline const &>(),
    std::declval<vkexec::resource_table const &>(),
    std::declval<vkexec::heap_table_lower_env>(),
    heap_push{}))>);
static_assert(vkexec::detail::sender_adaptor_closure<decltype(vkexec::bind_resources(vkexec::descriptor_heap,
    std::declval<vkexec::handles::compute_pipeline const &>(),
    std::declval<vkexec::resource_table const &>(),
    std::declval<vkexec::heap_table_lower_env>(),
    std::declval<fixed_heap_byte_span>()))>);

}// namespace

TEST_CASE("compute_pipeline can create a descriptor-heap null layout", "[vkexec][descriptor_heap][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  vkexec::ext::configure<vkexec::ext::descriptor_heap>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto pipe = vkexec::test::sync_wait_value(vkexec::factory::make_compute_pipeline(vkexec::descriptor_heap,
    *ctx,
    k_heap_compute_glsl,
    vkexec::heap_layout_desc{ .specialization = {}, .local_size = vkexec::k_default_local_size },
    "heap.comp"));
  REQUIRE(pipe.resources().pipeline != VK_NULL_HANDLE);
  REQUIRE(pipe.resources().pipeline_layout == VK_NULL_HANDLE);
  REQUIRE(pipe.resources().set_layout == VK_NULL_HANDLE);
  REQUIRE(pipe.resources().descriptor_pool == VK_NULL_HANDLE);
}

TEST_CASE("descriptor-heap compute_pipeline accepts specialization constants", "[vkexec][descriptor_heap][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  vkexec::ext::configure<vkexec::ext::descriptor_heap>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto pipe = vkexec::test::sync_wait_value(vkexec::factory::make_compute_pipeline(vkexec::descriptor_heap,
    *ctx,
    k_heap_spec_glsl,
    vkexec::heap_layout_desc{ .specialization = { k_work_count }, .local_size = vkexec::k_default_local_size },
    "heap_spec.comp"));
  REQUIRE(pipe.resources().pipeline != VK_NULL_HANDLE);
}

TEST_CASE("compute_pass records push data for descriptor-heap pipelines", "[vkexec][descriptor_heap][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  vkexec::ext::configure<vkexec::ext::descriptor_heap>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto pipe = vkexec::test::sync_wait_value(vkexec::factory::make_compute_pipeline(vkexec::descriptor_heap,
    *ctx,
    k_heap_compute_glsl,
    vkexec::heap_layout_desc{ .specialization = {}, .local_size = vkexec::k_default_local_size },
    "heap_pass.comp"));

  heap_push const params{ .count = k_work_count };
  auto waited = vkexec::test::sync_wait_sender(stdexec::schedule(ctx->get_scheduler())
                                               | vkexec::compute_pass(vkexec::descriptor_heap,
                                                 pipe,
                                                 std::vector<vkexec::image_use>{},
                                                 std::vector<vkexec::buffer_use>{},
                                                 params,
                                                 k_work_count));
  REQUIRE(vkexec::test::sync_wait_completed(waited));
}

TEST_CASE("dispatch_compute builds descriptor-heap algorithm passes", "[vkexec][descriptor_heap][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  vkexec::ext::configure<vkexec::ext::descriptor_heap>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
  auto allocator = vkexec::test::require_allocator(*ctx);
  vkexec::algorithm const algo =
    vkexec::test::sync_wait_value(vkexec::factory::make_compute_pipeline(vkexec::descriptor_heap,
      *ctx,
      k_heap_compute_glsl,
      vkexec::heap_layout_desc{ .specialization = {}, .local_size = vkexec::k_default_local_size },
      "heap_dispatch.comp"));

  heap_push const params{ .count = k_work_count };
  auto waited = vkexec::test::sync_wait_sender(
    stdexec::schedule(ctx->get_scheduler())
    | vkexec::dispatch_compute(vkexec::descriptor_heap, algo, {}, {}, params, k_work_count));
  REQUIRE(vkexec::test::sync_wait_completed(waited));
}

TEST_CASE("descriptor-heap shader accesses explicitly declared storage", "[vkexec][descriptor_heap][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  vkexec::ext::configure<vkexec::ext::descriptor_heap>(requirements);
  bool const require_validation =
#ifdef VKEXEC_TEST_VALIDATION
    true;
#else
    false;
#endif
  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context(
    { .validation_layers = require_validation, .requirements = std::move(requirements) }));
  if (!vkexec::ext::available<vkexec::ext::descriptor_heap>(*ctx)) { SKIP("descriptor_heap unavailable"); }
  if (!ctx->capabilities().shader_untyped_pointers) { SKIP("shaderUntypedPointers unavailable"); }
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto layout_result = vkexec::query_descriptor_heap_layout(*ctx);
  REQUIRE(layout_result);
  auto const &layout = vkexec::expected_get(layout_result);
  auto storage = vkexec::test::sync_wait_value(vkexec::vma::factory::make_gpu_buffer(allocator,
    vkexec::vma::gpu_buffer_create_info{ .size = sizeof(std::uint32_t),
      .memory = vkexec::vma::gpu_buffer_memory::host_visible,
      .shader_device_address = true }));
  auto mapped_storage = storage.mapped();
  REQUIRE(mapped_storage.size() >= sizeof(std::uint32_t));
  std::uint32_t const initial = k_initial_heap_value;
  std::memcpy(mapped_storage.data(), &initial, sizeof(initial));
  REQUIRE(storage.flush());
  auto heap = vkexec::test::sync_wait_value(vkexec::factory::make_descriptor_heap_buffer(
    allocator, vkexec::descriptor_heap_byte_size(layout, k_single_workgroup), layout.resource_heap_alignment));
  auto const address = storage.device_address();
  REQUIRE(address);
  auto heap_bytes = heap.mapped();
  REQUIRE(heap_bytes.size() >= layout.buffer_descriptor_size);
  REQUIRE(vkexec::write_storage_buffer_descriptor(
    *ctx, *address, storage.size(), heap_bytes.first(layout.buffer_descriptor_size)));
  REQUIRE(heap.flush());
  auto pipe = vkexec::test::sync_wait_value(vkexec::factory::make_compute_pipeline(vkexec::descriptor_heap,
    *ctx,
    k_heap_write_glsl,
    vkexec::heap_layout_desc{ .specialization = {}, .local_size = { k_single_workgroup, 1, 1 } },
    "heap_write.comp"));
  auto const heap_address = heap.device_address();
  REQUIRE(heap_address);
  if (layout.resource_heap_alignment != 0) { REQUIRE(*heap_address % layout.resource_heap_alignment == 0); }
  VkDeviceAddress const bound_heap_address = *heap_address;
  VkDeviceSize const reserved_offset = layout.descriptor_stride * k_single_workgroup;
  VkDeviceSize const aligned_offset =
    layout.resource_heap_alignment == 0
      ? reserved_offset
      : ((reserved_offset + layout.resource_heap_alignment - 1) / layout.resource_heap_alignment)
          * layout.resource_heap_alignment;
  auto bind_heap = vkexec::custom_pass(
    vkexec::uses(), [&ctx, bound_heap_address, &heap, aligned_offset, &layout](VkCommandBuffer cmd) -> vkexec::status {
      return vkexec::cmd_bind_resource_heap(
        *ctx, cmd, bound_heap_address, heap.size(), aligned_offset, layout.min_resource_heap_reserved_range);
    });
  auto const access =
    vkexec::read_write(storage.handle(), VkDeviceSize{}, storage.size(), vkexec::buffer_usage::storage_compute);
  auto graph = stdexec::schedule(ctx->get_scheduler()) | bind_heap
               | vkexec::compute_pass(vkexec::descriptor_heap,
                 pipe,
                 std::vector<vkexec::image_use>{},
                 std::vector<vkexec::buffer_use>{ access },
                 k_single_workgroup)
               | vkexec::compute_pass(vkexec::descriptor_heap,
                 pipe,
                 std::vector<vkexec::image_use>{},
                 std::vector<vkexec::buffer_use>{ access },
                 k_single_workgroup);
  auto waited = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(waited));
  REQUIRE(storage.invalidate());
  std::uint32_t result{};
  std::memcpy(&result, mapped_storage.data(), sizeof(result));
  REQUIRE(result == k_expected_heap_value);
}

TEST_CASE("tagged create draws without owning pipeline", "[vkexec][descriptor_heap][gpu][execution]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  vkexec::ext::configure<vkexec::ext::descriptor_heap>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto layout_result = vkexec::query_descriptor_heap_layout(*ctx);
  REQUIRE(layout_result.has_value());
  auto const &layout = vkexec::expected_get(layout_result);
  auto storage = vkexec::test::sync_wait_value(vkexec::vma::factory::make_gpu_buffer(allocator,
    vkexec::vma::gpu_buffer_create_info{
      .size = k_storage_bytes,
      .memory = vkexec::vma::buffer_memory::device_local,
      .shader_device_address = true,
    }));
  auto heap = vkexec::test::sync_wait_value(vkexec::factory::make_descriptor_heap_buffer(
    allocator, vkexec::descriptor_heap_byte_size(layout, 1), layout.resource_heap_alignment));
  auto const table = vkexec::bindings(
    vkexec::resource_binding{ .slot = 0, .resource = vkexec::buffer_resource(storage.handle(), storage.size()) });
  std::array<std::uint32_t, 1> const indices{ 0 };
  vkexec::heap_table_lower_env const lower_env{ .resource_heap_bytes = heap.mapped(),
    .sampler_heap_bytes = {},
    .buffer_descriptor_size = layout.buffer_descriptor_size,
    .image_descriptor_size = layout.image_descriptor_size,
    .descriptor_stride = layout.descriptor_stride,
    .sampler_descriptor_size = layout.sampler_descriptor_size,
    .sampler_descriptor_stride = layout.sampler_descriptor_size,
    .indices = indices,
    .sampler_indices = {},
    .image_view_infos = {},
    .sampler_infos = {} };
  auto resources_result = vkexec::create(vkexec::descriptor_heap,
    *ctx,
    k_heap_compute_glsl,
    vkexec::heap_layout_desc{ .specialization = {}, .local_size = vkexec::k_default_local_size },
    "heap_execution.comp");
  REQUIRE(resources_result.has_value());
  auto resources = vkexec::expected_take(resources_result);
  REQUIRE(resources.pipeline != VK_NULL_HANDLE);
  REQUIRE(resources.pipeline_layout == VK_NULL_HANDLE);
  REQUIRE(resources.set_layout == VK_NULL_HANDLE);
  REQUIRE(resources.descriptor_pool == VK_NULL_HANDLE);

  heap_push const params{ .count = k_work_count };
  auto const groups = vkexec::groups_for(resources, k_work_count);
  auto graph = stdexec::schedule(ctx->get_scheduler())
               | vkexec::bind_resources(vkexec::descriptor_heap, resources, table, lower_env, params)
               | vkexec::compute_pass(vkexec::bind_compute(resources).with_resources({}, {}), groups);
  auto waited = vkexec::test::sync_wait_sender(std::move(graph));
  REQUIRE(vkexec::test::sync_wait_completed(waited));

  vkexec::destroy(*ctx, resources);
}

TEST_CASE("create builds null-layout DR pipeline", "[vkexec][descriptor_heap][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  vkexec::ext::configure<vkexec::ext::descriptor_heap>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto resources_result = vkexec::create(vkexec::descriptor_heap,
    *ctx,
    vkexec::shaders::k_triangle_vert,
    vkexec::shaders::k_triangle_frag,
    vkexec::heap_graphics_layout_desc{
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .blend = vkexec::blend_mode::premultiplied,
      .depth_test = false,
      .depth_write = false,
      .color_formats = { VK_FORMAT_R8G8B8A8_UNORM },
      .depth_format = VK_FORMAT_UNDEFINED,
    },
    "heap_gfx.vert",
    "heap_gfx.frag");
  REQUIRE(resources_result.has_value());
  auto resources = vkexec::expected_take(resources_result);
  REQUIRE(resources.pipeline != VK_NULL_HANDLE);
  REQUIRE(resources.pipeline_layout == VK_NULL_HANDLE);
  REQUIRE(resources.set_layout == VK_NULL_HANDLE);
  REQUIRE(resources.descriptor_pool == VK_NULL_HANDLE);
  REQUIRE(resources.shader == VK_NULL_HANDLE);

  vkexec::destroy(vkexec::descriptor_heap, *ctx, resources);
  REQUIRE(resources.pipeline == VK_NULL_HANDLE);
}

TEST_CASE("graphics_pipeline tag factory owns null-layout DR pipeline", "[vkexec][descriptor_heap][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  vkexec::ext::configure<vkexec::ext::descriptor_heap>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto pipe = vkexec::test::sync_wait_value(vkexec::factory::make_graphics_pipeline(vkexec::descriptor_heap,
    *ctx,
    vkexec::shaders::k_triangle_vert,
    vkexec::shaders::k_triangle_frag,
    vkexec::heap_graphics_layout_desc{
      .blend = vkexec::blend_mode::premultiplied,
      .depth_test = false,
      .depth_write = false,
      .color_formats = { VK_FORMAT_R8G8B8A8_UNORM },
    },
    "heap_own.vert",
    "heap_own.frag"));
  REQUIRE(pipe.resources().pipeline != VK_NULL_HANDLE);
  REQUIRE(pipe.bind().pipeline == pipe.resources().pipeline);
  REQUIRE(pipe.bind().layout == VK_NULL_HANDLE);
  REQUIRE(pipe.bind().set == VK_NULL_HANDLE);
}

// NOLINTEND(readability-function-cognitive-complexity)
