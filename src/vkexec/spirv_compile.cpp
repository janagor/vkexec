#include <vkexec/spirv_compile.hpp>

#include <vkexec/error.hpp>
#include <vkexec/result.hpp>

#include <glslang/Include/ResourceLimits.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace vkexec {
namespace {

  constexpr int k_vulkan_glsl_dialect_version = 100;
  constexpr int k_glsl_version_450 = 450;
  constexpr int k_glsl_version_460 = 460;

  struct glslang_targets
  {
    glslang::EShTargetClientVersion vulkan_client{ glslang::EShTargetVulkan_1_0 };
    glslang::EShTargetLanguageVersion spirv{ glslang::EShTargetSpv_1_0 };
    int default_glsl_version{ k_glsl_version_450 };
  };

  auto targets_for_vulkan_api(std::uint32_t vulkan_api_version) -> glslang_targets
  {
    // Match SPIR-V / GLSL dialect to the negotiated device API so newer builtins compile.
    auto const major = VK_API_VERSION_MAJOR(vulkan_api_version);
    auto const minor = VK_API_VERSION_MINOR(vulkan_api_version);
    if (major < 1) { return {}; }

    glslang_targets targets{};
    if (minor >= 4) {
      targets.vulkan_client = glslang::EShTargetVulkan_1_4;
      targets.spirv = glslang::EShTargetSpv_1_6;
      targets.default_glsl_version = k_glsl_version_460;
    } else if (minor >= 3) {
      targets.vulkan_client = glslang::EShTargetVulkan_1_3;
      targets.spirv = glslang::EShTargetSpv_1_6;
      targets.default_glsl_version = k_glsl_version_460;
    } else if (minor >= 2) {
      targets.vulkan_client = glslang::EShTargetVulkan_1_2;
      targets.spirv = glslang::EShTargetSpv_1_5;
      targets.default_glsl_version = k_glsl_version_450;
    } else if (minor >= 1) {
      targets.vulkan_client = glslang::EShTargetVulkan_1_1;
      targets.spirv = glslang::EShTargetSpv_1_3;
      targets.default_glsl_version = k_glsl_version_450;
    } else {
      targets.vulkan_client = glslang::EShTargetVulkan_1_0;
      targets.spirv = glslang::EShTargetSpv_1_0;
      targets.default_glsl_version = k_glsl_version_450;
    }
    return targets;
  }

  auto ensure_glslang() -> void
  {
    // Process-wide init; glslang is not safe to InitializeProcess repeatedly.
    // NOLINTNEXTLINE(misc-const-correctness)
    static std::once_flag once;
    std::call_once(once, []() -> void { glslang::InitializeProcess(); });
  }

  auto to_glslang(shader_kind kind) -> EShLanguage
  {
    switch (kind) {
    case shader_kind::vertex:
      return EShLangVertex;
    case shader_kind::fragment:
      return EShLangFragment;
    case shader_kind::compute:
    default:
      return EShLangCompute;
    }
  }

}// namespace

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto compile_glsl_to_spirv(std::string_view glsl_source,
  std::string_view name,
  shader_kind kind,
  std::uint32_t vulkan_api_version) -> result<std::vector<std::uint32_t>>
{
  ensure_glslang();

  auto const targets = targets_for_vulkan_api(vulkan_api_version);
  EShLanguage const stage = to_glslang(kind);
  glslang::TShader shader(stage);
  std::array<char const *, 1> strings{ { glsl_source.data() } };
  std::array<int, 1> lengths{ { static_cast<int>(glsl_source.size()) } };
  shader.setStringsWithLengths(strings.data(), lengths.data(), 1);
  shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, k_vulkan_glsl_dialect_version);
  shader.setEnvClient(glslang::EShClientVulkan, targets.vulkan_client);
  shader.setEnvTarget(glslang::EShTargetSpv, targets.spirv);

  TBuiltInResource const &resources = *GetDefaultResources();
  // NOLINTNEXTLINE(hicpp-signed-bitwise,clang-analyzer-optin.core.EnumCastOutOfRange)
  auto const messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
  if (!shader.parse(&resources, targets.default_glsl_version, false, messages)) {
    std::string detail = "glslang parse failed for ";
    detail += name;
    detail += ":\n";
    detail += shader.getInfoLog();
    return fail(errc::parse_error, detail);
  }

  glslang::TProgram program;
  program.addShader(&shader);
  if (!program.link(messages)) {
    std::string detail = "glslang link failed for ";
    detail += name;
    detail += ":\n";
    detail += program.getInfoLog();
    return fail(errc::parse_error, detail);
  }

  std::vector<std::uint32_t> spirv;
  glslang::SpvOptions options;
  options.generateDebugInfo = false;
  options.disableOptimizer = false;
  options.optimizeSize = false;
  glslang::GlslangToSpv(*program.getIntermediate(stage), spirv, &options);
  if (spirv.empty()) { return fail(errc::empty_result, "SPIR-V emission produced empty module"); }
  return spirv;
}

}// namespace vkexec
