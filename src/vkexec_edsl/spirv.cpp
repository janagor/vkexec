#include <vkexec_edsl/spirv.hpp>

#include <glslang/Include/ResourceLimits.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <array>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace vkexec::edsl {
namespace {

  constexpr int k_glslang_vulkan_client_version = 100;
  constexpr int k_glsl_version = 450;

  auto ensure_glslang() -> void
  {
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
auto compile_glsl_to_spirv(std::string_view glsl_source, std::string_view name, shader_kind kind)
  -> std::vector<std::uint32_t>
{
  ensure_glslang();

  EShLanguage const stage = to_glslang(kind);
  glslang::TShader shader(stage);
  std::array<char const *, 1> strings{ { glsl_source.data() } };
  std::array<int, 1> lengths{ { static_cast<int>(glsl_source.size()) } };
  shader.setStringsWithLengths(strings.data(), lengths.data(), 1);
  shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, k_glslang_vulkan_client_version);
  shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);
  shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_5);

  TBuiltInResource const &resources = *GetDefaultResources();
  // NOLINTNEXTLINE(hicpp-signed-bitwise,clang-analyzer-optin.core.EnumCastOutOfRange)
  auto const messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
  if (!shader.parse(&resources, k_glsl_version, false, messages)) {
    throw std::runtime_error(
      std::string("glslang parse failed for ") + std::string(name) + ":\n" + shader.getInfoLog());
  }

  glslang::TProgram program;
  program.addShader(&shader);
  if (!program.link(messages)) {
    throw std::runtime_error(
      std::string("glslang link failed for ") + std::string(name) + ":\n" + program.getInfoLog());
  }

  std::vector<std::uint32_t> spirv;
  glslang::SpvOptions options;
  options.generateDebugInfo = false;
  options.disableOptimizer = false;
  options.optimizeSize = false;
  glslang::GlslangToSpv(*program.getIntermediate(stage), spirv, &options);
  if (spirv.empty()) { throw std::runtime_error("SPIR-V emission produced empty module"); }
  return spirv;
}

}// namespace vkexec::edsl
