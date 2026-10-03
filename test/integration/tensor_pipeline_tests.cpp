#include "test_helpers.hpp"
#include "vma_test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/tensor.hpp>
#include <vkexec/tensor_pass.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_features/buffer_device_address.hpp>
#include <vkexec_features/feature.hpp>

#include <stdexec/execution.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

namespace ex = stdexec;

namespace {

constexpr std::size_t k_count = 8;
constexpr float k_fill = 1.5F;
constexpr std::uint32_t k_local_size = 8;

constexpr std::string_view k_scale_glsl = R"(
#version 450
layout(local_size_x = 8) in;
layout(set = 0, binding = 0) buffer Values { float data[]; } values;
layout(push_constant) uniform Push { float scale; } pc;
void main() {
  values.data[gl_GlobalInvocationID.x] *= pc.scale;
}
)";

struct scale_params
{
  float scale;
};

[[nodiscard]] auto require_tensor_context() -> std::unique_ptr<vkexec::context>
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 2;
  vkexec::feat::configure<vkexec::feat::buffer_device_address>(requirements);
  return vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
}

}// namespace

TEST_CASE("tensor_pass uploads, dispatches, and downloads tensors", "[vkexec][tensor][gpu]")
{
  auto ctx = require_tensor_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto values = vkexec::test::sync_wait_value(vkexec::factory::make_tensor(allocator, k_count, k_fill));

  auto resources_result = vkexec::create(*ctx,
    k_scale_glsl,
    vkexec::layout_desc{
      .binding_kinds = {},
      .binding_slots = {},
      .bindings = { vkexec::buffer_access::readwrite },
      .push_constant_size = sizeof(scale_params),
      .specialization = {},
      .local_size = { k_local_size, 1, 1 },
    },
    "tensor_pass.comp");
  REQUIRE(resources_result);
  auto resources = vkexec::expected_take(resources_result);

  std::array const bindings{ values.storage_binding(0) };
  auto bound_result = vkexec::bind_storage(*ctx, resources, bindings);
  REQUIRE(bound_result);
  auto bound = vkexec::expected_take(bound_result);

  scale_params const params{ .scale = 2.0F };
  auto completed = vkexec::test::sync_wait_sender(
    ex::schedule(ctx->get_scheduler())
    | vkexec::tensor_pass(resources, bound.set, params, static_cast<std::uint32_t>(k_count), values));
  REQUIRE(vkexec::test::sync_wait_completed(completed));
  REQUIRE(values.span().front() == k_fill * params.scale);
  REQUIRE(values.span().back() == k_fill * params.scale);

  vkexec::free_compute_set(*ctx, resources, bound.set);
  vkexec::destroy(*ctx, resources);
}
