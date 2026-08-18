#ifndef VKEXEC_COMPUTE_PIPELINE_HPP
#define VKEXEC_COMPUTE_PIPELINE_HPP

#include <vkexec/context.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/push.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <span>
#include <string_view>

namespace vkexec {

struct storage_binding
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  VkDeviceSize byte_size{ 0 };
};

/// Handle to a cached compute pipeline built from existing SPIR-V (hybrid / embed path).
class compute_pipeline
{
public:
  [[nodiscard]] static auto from_spirv(context &ctx, std::span<std::uint32_t const> spirv, layout_desc const &desc)
    -> compute_pipeline;
  [[nodiscard]] static auto
    from_glsl(context &ctx, std::string_view glsl, layout_desc const &desc, std::string_view name = "vkexec.comp")
      -> compute_pipeline;

  [[nodiscard]] auto resources() noexcept -> pipeline_resources & { return *resources_; }
  [[nodiscard]] auto resources() const noexcept -> pipeline_resources const & { return *resources_; }

  [[nodiscard]] auto bind(VkDescriptorSet set = VK_NULL_HANDLE) const -> compute_bind
  { return bind_compute(*resources_, set); }

  [[nodiscard]] auto local_size() const noexcept -> dispatch
  {
    return dispatch{
      .x = resources_->local_size.at(0), .y = resources_->local_size.at(1), .z = resources_->local_size.at(2)
    };
  }

  [[nodiscard]] auto groups_for(std::uint32_t work_count) const noexcept -> dispatch
  {
    std::uint32_t const local_x = resources_->local_size.at(0) == 0U ? 1U : resources_->local_size.at(0);
    return dispatch{ .x = (work_count + local_x - 1U) / local_x };
  }

  [[nodiscard]] auto allocate_set() -> VkDescriptorSet;
  auto update_set(VkDescriptorSet set, std::span<storage_binding const> buffers) -> void;

private:
  compute_pipeline(context *ctx, pipeline_resources *pipe) noexcept : ctx_(ctx), resources_(pipe) {}

  context *ctx_{ nullptr };
  pipeline_resources *resources_{ nullptr };
};

template<typename Params>
auto compute_pass(compute_pipeline const &pipe, VkDescriptorSet set, Params const &params, std::uint32_t work_count)
  -> prebuilt_compute_pass_closure
{ return compute_pass(pipe.bind(set), params, pipe.groups_for(work_count)); }

inline auto compute_pass(compute_pipeline const &pipe, VkDescriptorSet set, std::uint32_t work_count)
  -> prebuilt_compute_pass_closure
{ return compute_pass(pipe.bind(set), pipe.groups_for(work_count)); }

template<typename T>
auto upload_push_constants(VkCommandBuffer cmd, compute_pipeline const &pipe, T const &params) -> void
{ upload_push_constants(cmd, pipe.resources().pipeline_layout, params); }

}// namespace vkexec

#endif// VKEXEC_COMPUTE_PIPELINE_HPP
