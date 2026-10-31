#ifndef VKEXEC_TEST_RECORDING_COMPUTE_HPP
#define VKEXEC_TEST_RECORDING_COMPUTE_HPP
#include <memory>
#include <array>
#include <cstdint>
#include <vkexec/context.hpp>
#include <vkexec/result.hpp>
#include <vulkan/vulkan_core.h>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>

namespace vkexec::test {
// Generated from recording.comp with glslangValidator -V.
inline constexpr std::array<std::uint32_t, 200> k_recording_spirv{
  0x07230203U,
  0x00010000U,
  0x0008000bU,
  0x0000001dU,
  0x00000000U,
  0x00020011U,
  0x00000001U,
  0x0006000bU,
  0x00000001U,
  0x4c534c47U,
  0x6474732eU,
  0x3035342eU,
  0x00000000U,
  0x0003000eU,
  0x00000000U,
  0x00000001U,
  0x0006000fU,
  0x00000005U,
  0x00000004U,
  0x6e69616dU,
  0x00000000U,
  0x00000010U,
  0x00060010U,
  0x00000004U,
  0x00000011U,
  0x00000001U,
  0x00000001U,
  0x00000001U,
  0x00030003U,
  0x00000002U,
  0x000001c2U,
  0x00040005U,
  0x00000004U,
  0x6e69616dU,
  0x00000000U,
  0x00040005U,
  0x00000008U,
  0x61746144U,
  0x00000000U,
  0x00050006U,
  0x00000008U,
  0x00000000U,
  0x756c6176U,
  0x00007365U,
  0x00040005U,
  0x0000000aU,
  0x61746164U,
  0x00000000U,
  0x00080005U,
  0x00000010U,
  0x475f6c67U,
  0x61626f6cU,
  0x766e496cU,
  0x7461636fU,
  0x496e6f69U,
  0x00000044U,
  0x00040047U,
  0x00000007U,
  0x00000006U,
  0x00000004U,
  0x00030047U,
  0x00000008U,
  0x00000003U,
  0x00050048U,
  0x00000008U,
  0x00000000U,
  0x00000023U,
  0x00000000U,
  0x00040047U,
  0x0000000aU,
  0x00000021U,
  0x00000000U,
  0x00040047U,
  0x0000000aU,
  0x00000022U,
  0x00000000U,
  0x00040047U,
  0x00000010U,
  0x0000000bU,
  0x0000001cU,
  0x00040047U,
  0x0000001cU,
  0x0000000bU,
  0x00000019U,
  0x00020013U,
  0x00000002U,
  0x00030021U,
  0x00000003U,
  0x00000002U,
  0x00040015U,
  0x00000006U,
  0x00000020U,
  0x00000000U,
  0x0003001dU,
  0x00000007U,
  0x00000006U,
  0x0003001eU,
  0x00000008U,
  0x00000007U,
  0x00040020U,
  0x00000009U,
  0x00000002U,
  0x00000008U,
  0x0004003bU,
  0x00000009U,
  0x0000000aU,
  0x00000002U,
  0x00040015U,
  0x0000000bU,
  0x00000020U,
  0x00000001U,
  0x0004002bU,
  0x0000000bU,
  0x0000000cU,
  0x00000000U,
  0x0004002bU,
  0x00000006U,
  0x0000000dU,
  0x00000003U,
  0x00040017U,
  0x0000000eU,
  0x00000006U,
  0x00000003U,
  0x00040020U,
  0x0000000fU,
  0x00000001U,
  0x0000000eU,
  0x0004003bU,
  0x0000000fU,
  0x00000010U,
  0x00000001U,
  0x0004002bU,
  0x00000006U,
  0x00000011U,
  0x00000000U,
  0x00040020U,
  0x00000012U,
  0x00000001U,
  0x00000006U,
  0x0004002bU,
  0x00000006U,
  0x00000016U,
  0x00000001U,
  0x00040020U,
  0x00000017U,
  0x00000002U,
  0x00000006U,
  0x0006002cU,
  0x0000000eU,
  0x0000001cU,
  0x00000016U,
  0x00000016U,
  0x00000016U,
  0x00050036U,
  0x00000002U,
  0x00000004U,
  0x00000000U,
  0x00000003U,
  0x000200f8U,
  0x00000005U,
  0x00050041U,
  0x00000012U,
  0x00000013U,
  0x00000010U,
  0x00000011U,
  0x0004003dU,
  0x00000006U,
  0x00000014U,
  0x00000013U,
  0x00050080U,
  0x00000006U,
  0x00000015U,
  0x0000000dU,
  0x00000014U,
  0x00060041U,
  0x00000017U,
  0x00000018U,
  0x0000000aU,
  0x0000000cU,
  0x00000015U,
  0x0004003dU,
  0x00000006U,
  0x00000019U,
  0x00000018U,
  0x00050080U,
  0x00000006U,
  0x0000001aU,
  0x00000019U,
  0x00000016U,
  0x00060041U,
  0x00000017U,
  0x0000001bU,
  0x0000000aU,
  0x0000000cU,
  0x00000015U,
  0x0003003eU,
  0x0000001bU,
  0x0000001aU,
  0x000100fdU,
  0x00010038U,
};
struct recording_compute
{
  context *ctx{};
  handles::compute_pipeline pipeline;
  VkDescriptorSet set{};
  compute_bind bind;
  recording_compute() = default;
  recording_compute(recording_compute const &) = delete;
  auto operator=(recording_compute const &) -> recording_compute & = delete;
  recording_compute(recording_compute &&) = delete;
  auto operator=(recording_compute &&) -> recording_compute & = delete;
  ~recording_compute()
  {
    if (set != VK_NULL_HANDLE) { free_compute_set(*ctx, pipeline, set); }
    if (ctx != nullptr) { destroy(*ctx, pipeline); }
  }
  [[nodiscard]] static auto make(context &ctx, VkBuffer buffer, VkDeviceSize size)
    -> result<std::unique_ptr<recording_compute>>
  {
    auto fixture = std::make_unique<recording_compute>();
    fixture->ctx = &ctx;
    layout_desc layout{};
    layout.bindings = { buffer_access::readwrite };
    layout.local_size = { 1, 1, 1 };
    auto pipeline = create(ctx, k_recording_spirv, layout);
    if (!pipeline) { return fail(pipeline); }
    fixture->pipeline = expected_take(pipeline);
    std::array const bindings{ storage_binding{ .buffer = buffer, .byte_size = size, .binding = 0 } };
    auto bound = bind_storage(ctx, fixture->pipeline, bindings);
    if (!bound) { return fail(bound); }
    fixture->set = bound->set;
    fixture->bind = bind_compute(fixture->pipeline, fixture->set)
                      .with_resources({},
                        { buffer_use{ .buffer = buffer,
                          .offset = 0,
                          .size = size,
                          .usage = buffer_usage::storage_compute,
                          .access = resource_access::read_write } });
    return fixture;
  }
};
}// namespace vkexec::test
#endif
