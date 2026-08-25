#include <catch2/catch_test_macros.hpp>

#include <vkexec/compute_pipeline.hpp>
#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace ex = stdexec;

namespace {

constexpr std::string_view k_heap_compute_glsl = R"(#version 460
layout(local_size_x = 64) in;
void main() {}
)";

struct heap_push
{
  std::uint32_t count;
};

auto skip_if_unavailable(std::exception const &error) -> void
{ SKIP(std::string("Vulkan feature set unavailable: ") + error.what()); }

}// namespace

TEST_CASE("compute_pipeline can create a descriptor-heap null layout", "[vkexec][descriptor_heap][gpu]")
{
  VkPhysicalDeviceDescriptorHeapFeaturesEXT features_heap{};
  features_heap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
  features_heap.descriptorHeap = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  requirements.require_extension_feature(features_heap);

  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(vkexec::scheduler_options{ .requirements = std::move(requirements) }); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_unavailable(error); }

  auto pipe = vkexec::compute_pipeline::from_glsl(*ctx,
    k_heap_compute_glsl,
    vkexec::layout_desc{
      .bindings = {},
      .push_constant_size = 0,
      .specialization = {},
      .local_size = vkexec::k_default_local_size,
      .descriptor_heap = true,
    },
    "heap.comp");
  REQUIRE(pipe.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  REQUIRE(pipe->resources().pipeline != VK_NULL_HANDLE);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  REQUIRE(pipe->resources().pipeline_layout == VK_NULL_HANDLE);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  REQUIRE(pipe->resources().set_layout == VK_NULL_HANDLE);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  REQUIRE(pipe->resources().descriptor_pool == VK_NULL_HANDLE);
}

TEST_CASE("compute_pass records bindless push data for heap pipelines", "[vkexec][descriptor_heap][gpu]")
{
  VkPhysicalDeviceDescriptorHeapFeaturesEXT features_heap{};
  features_heap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
  features_heap.descriptorHeap = VK_TRUE;

  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  requirements.device_extensions = { VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME };
  requirements.require_extension_feature(features_heap);

  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(vkexec::scheduler_options{ .requirements = std::move(requirements) }); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_unavailable(error); }

  auto pipe = vkexec::compute_pipeline::from_glsl(*ctx,
    k_heap_compute_glsl,
    vkexec::layout_desc{
      .bindings = {},
      .push_constant_size = 0,
      .specialization = {},
      .local_size = vkexec::k_default_local_size,
      .descriptor_heap = true,
    },
    "heap_pass.comp");
  REQUIRE(pipe.has_value());

  heap_push const params{ .count = 64 };
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  auto result = ex::sync_wait(ex::schedule(ctx->get_scheduler()) | vkexec::compute_pass(*pipe, params, 64U));
  REQUIRE(result.has_value());
}
