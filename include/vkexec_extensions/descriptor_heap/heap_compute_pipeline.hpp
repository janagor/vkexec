#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_COMPUTE_PIPELINE_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_COMPUTE_PIPELINE_HPP

//! \file
//! Bindless compute pipelines (`VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT`).

#include <vkexec/compute_pipeline.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec_extensions/descriptor_heap/strategy.hpp>

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace vkexec {

/**
 * Layout for bindless null-layout compute pipelines.
 *
 * No descriptor sets or classic push constants; parameters use `cmd_push_data`.
 *
 * @see create, compute_pipeline
 */
struct heap_layout_desc
{
  std::vector<std::uint32_t> specialization;
  std::array<std::uint32_t, 3> local_size{ k_default_local_size };
};

static_assert(std::is_nothrow_move_constructible_v<heap_layout_desc>);

/**
 * Creates bindless heap compute Vulkan objects from SPIR-V.
 *
 * Returns a `handles::compute_pipeline` bag with null `set_layout` / `pipeline_layout` /
 * `descriptor_pool`. Do not call classic `bind_storage` on these bags. Caller owns
 * the handles and must call `destroy`.
 */
[[nodiscard]] auto
  create(descriptor_heap_t strategy, context &ctx, std::span<std::uint32_t const> spirv, heap_layout_desc const &desc)
    -> result<handles::compute_pipeline>;

/**
 * Compiles `glsl` then creates bindless heap compute Vulkan objects.
 *
 * @param name Debug name for the compiler.
 */
[[nodiscard]] auto create(descriptor_heap_t strategy,
  context &ctx,
  std::string_view glsl,
  heap_layout_desc const &desc,
  std::string_view name = "heap.comp") -> result<handles::compute_pipeline>;

namespace detail {

  struct create_heap_compute_pipeline_spirv_factory
  {
    descriptor_heap_t strategy;
    context *ctx;
    std::vector<std::uint32_t> spirv;
    heap_layout_desc desc;

    [[nodiscard]] auto operator()() const -> result<owned::compute_pipeline>;
  };

  struct create_heap_compute_pipeline_glsl_factory
  {
    descriptor_heap_t strategy;
    context *ctx;
    std::string glsl;
    heap_layout_desc desc;
    std::string name;

    [[nodiscard]] auto operator()() const -> result<owned::compute_pipeline>;
  };

}// namespace detail

//! Owning factory customization used by `factory::make_compute_pipeline(descriptor_heap, ...)`.
[[nodiscard]] inline auto create_compute_pipeline(descriptor_heap_t strategy,
  context &ctx,
  std::span<std::uint32_t const> spirv,
  heap_layout_desc const &desc)
{
  return make_sender(detail::create_heap_compute_pipeline_spirv_factory{
    .strategy = strategy, .ctx = &ctx, .spirv = std::vector<std::uint32_t>(spirv.begin(), spirv.end()), .desc = desc });
}

//! Owning GLSL factory customization used by `factory::make_compute_pipeline(descriptor_heap, ...)`.
[[nodiscard]] inline auto create_compute_pipeline(descriptor_heap_t strategy,
  context &ctx,
  std::string_view glsl,
  heap_layout_desc const &desc,
  std::string_view name = "heap.comp")
{
  return make_sender(detail::create_heap_compute_pipeline_glsl_factory{
    .strategy = strategy, .ctx = &ctx, .glsl = std::string{ glsl }, .desc = desc, .name = std::string{ name } });
}

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_HEAP_COMPUTE_PIPELINE_HPP
