#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>

#include <vkexec/context.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/result.hpp>
#include <vkexec/spirv_compile.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vkexec {
namespace {

  constexpr std::size_t k_spirv_cache_tag = 0xBEEF51B5ULL;
  constexpr std::size_t k_hash_golden_ratio = 0x9e3779b97f4a7c15ULL;
  constexpr std::size_t k_hash_shift_left = 6U;
  constexpr std::size_t k_hash_shift_right = 2U;

  auto hash_combine(std::size_t seed, std::size_t value) -> std::size_t
  { return seed ^ (value + k_hash_golden_ratio + (seed << k_hash_shift_left) + (seed >> k_hash_shift_right)); }

  auto hash_spirv_heap_layout(std::span<std::uint32_t const> spirv, heap_layout_desc const &desc) -> std::size_t
  {
    std::size_t hash = k_spirv_cache_tag;
    hash = hash_combine(hash, spirv.size());
    // cppcheck-suppress useStlAlgorithm
    for (std::uint32_t const word : spirv) { hash = hash_combine(hash, word); }
    hash = hash_combine(hash, desc.specialization.size());
    // cppcheck-suppress useStlAlgorithm
    for (std::uint32_t const value : desc.specialization) { hash = hash_combine(hash, value); }
    hash = hash_combine(hash, desc.local_size.at(0));
    hash = hash_combine(hash, desc.local_size.at(1));
    hash = hash_combine(hash, desc.local_size.at(2));
    return hash;
  }

  auto destroy_heap_resources(context const &ctx, pipeline_resources &resources) -> void
  {
    VkDevice device = ctx.device();
    if (resources.pipeline != VK_NULL_HANDLE) { vkDestroyPipeline(device, resources.pipeline, nullptr); }
    if (resources.shader != VK_NULL_HANDLE) { vkDestroyShaderModule(device, resources.shader, nullptr); }
  }

  auto create_shader_module(VkDevice device, std::span<std::uint32_t const> spirv) -> result<VkShaderModule>
  {
    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = spirv.size_bytes();
    module_info.pCode = spirv.data();
    VkShaderModule shader{ VK_NULL_HANDLE };
    VkResult const create_result = vkCreateShaderModule(device, &module_info, nullptr, &shader);
    if (create_result != VK_SUCCESS) { return fail(create_result, "vkCreateShaderModule failed"); }
    return shader;
  }

  auto create_heap_compute_pipeline(VkDevice device, VkShaderModule shader, VkSpecializationInfo const *specialization)
    -> result<VkPipeline>
  {
    VkPipelineCreateFlags2CreateInfo flags2{};
    flags2.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO;
    flags2.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
    VkComputePipelineCreateInfo compute_info{};
    compute_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_info.pNext = &flags2;
    compute_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compute_info.stage.module = shader;
    compute_info.stage.pName = "main";
    compute_info.stage.pSpecializationInfo = specialization;
    compute_info.layout = VK_NULL_HANDLE;
    VkPipeline pipeline{ VK_NULL_HANDLE };
    VkResult const create_result =
      vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compute_info, nullptr, &pipeline);
    if (create_result != VK_SUCCESS) { return fail(create_result, "vkCreateComputePipelines failed"); }
    return pipeline;
  }

  class heap_pipeline_cache
  {
  public:
    explicit heap_pipeline_cache(context &ctx) : ctx_(&ctx) {}

    auto get_or_create(std::span<std::uint32_t const> spirv, heap_layout_desc const &desc)
      -> result<std::reference_wrapper<pipeline_resources>>
    {
      if (spirv.empty()) { return fail(errc::invalid_argument, "heap_compute_pipeline requires non-empty SPIR-V"); }

      std::size_t const key = hash_spirv_heap_layout(spirv, desc);
      {
        std::scoped_lock const lock(mutex_);
        if (auto cached = cache_.find(key); cached != cache_.end()) { return std::ref(*cached->second); }
      }

      auto resources = std::make_unique<pipeline_resources>();
      resources->local_size = desc.local_size;

      // Same constantID = index convention as classic compute_pipeline / pipeline_cache.
      std::vector<VkSpecializationMapEntry> spec_entries(desc.specialization.size());
      for (std::size_t index = 0; index < desc.specialization.size(); ++index) {
        spec_entries.at(index).constantID = static_cast<std::uint32_t>(index);
        spec_entries.at(index).offset = static_cast<std::uint32_t>(index * sizeof(std::uint32_t));
        spec_entries.at(index).size = sizeof(std::uint32_t);
      }
      VkSpecializationInfo spec_info{};
      VkSpecializationInfo const *spec_ptr = nullptr;
      if (!desc.specialization.empty()) {
        spec_info.mapEntryCount = static_cast<std::uint32_t>(spec_entries.size());
        spec_info.pMapEntries = spec_entries.data();
        spec_info.dataSize = desc.specialization.size() * sizeof(std::uint32_t);
        spec_info.pData = desc.specialization.data();
        spec_ptr = &spec_info;
      }

      VkDevice device = ctx_->device();
      auto shader_result = create_shader_module(device, spirv);
      if (!shader_result) { return fail(shader_result); }
      resources->shader = expected_take(shader_result);

      auto pipeline_result = create_heap_compute_pipeline(device, resources->shader, spec_ptr);
      if (!pipeline_result) {
        destroy_heap_resources(*ctx_, *resources);
        return fail(pipeline_result);
      }
      resources->pipeline = expected_take(pipeline_result);

      std::scoped_lock const lock(mutex_);
      if (auto cached = cache_.find(key); cached != cache_.end()) {
        destroy_heap_resources(*ctx_, *resources);
        return std::ref(*cached->second);
      }
      auto const inserted = cache_.emplace(key, std::move(resources));
      return std::ref(*inserted.first->second);
    }

  private:
    context *ctx_{ nullptr };
    std::mutex mutex_;
    std::unordered_map<std::size_t, std::unique_ptr<pipeline_resources>> cache_;
  };

  auto heap_cache_for(context &ctx) -> heap_pipeline_cache &
  {
    static std::mutex mutex;
    static std::unordered_map<context *, std::unique_ptr<heap_pipeline_cache>> caches;
    std::scoped_lock const lock(mutex);
    auto const found = caches.find(&ctx);
    if (found != caches.end()) { return *found->second; }
    auto inserted = caches.emplace(&ctx, std::make_unique<heap_pipeline_cache>(ctx));
    return *inserted.first->second;
  }

}// namespace

auto heap_compute_pipeline::create(context &ctx, std::span<std::uint32_t const> spirv, heap_layout_desc const &desc)
  -> detail::sync_sender_fn<heap_compute_pipeline>
{
  return detail::make_sync_sender_fn<heap_compute_pipeline>([&ctx, spirv, desc]() -> result<heap_compute_pipeline> {
    VKEXEC_TRY_ASSIGN(cached, heap_cache_for(ctx).get_or_create(spirv, desc));
    return heap_compute_pipeline{ nullptr, &cached.get() };
  });
}

auto heap_compute_pipeline::create(context &ctx,
  std::string_view glsl,
  heap_layout_desc const &desc,
  std::string_view name) -> detail::sync_sender_fn<heap_compute_pipeline>
{
  return detail::make_sync_sender_fn<heap_compute_pipeline>(
    [&ctx, glsl = std::string(glsl), desc, name = std::string(name)]() -> result<heap_compute_pipeline> {
      if (glsl.empty()) { return fail(errc::invalid_argument, "heap_compute_pipeline requires non-empty GLSL"); }
      VKEXEC_TRY_ASSIGN(spirv, compile_glsl_to_spirv(glsl, name, shader_kind::compute, ctx.api_version()));
      VKEXEC_TRY_ASSIGN(cached, heap_cache_for(ctx).get_or_create(spirv, desc));
      return heap_compute_pipeline{ nullptr, &cached.get() };
    });
}

}// namespace vkexec
