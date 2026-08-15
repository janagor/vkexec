#include <vkexec/detail/spirv.hpp>

#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <mutex>
#include <stdexcept>
#include <string>

namespace vkexec {
namespace {

void ensure_glslang()
{
  static std::once_flag once;
  std::call_once(once, [] { glslang::InitializeProcess(); });
}

EShLanguage to_glslang(shader_kind kind)
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

} // namespace

std::vector<std::uint32_t> compile_glsl_to_spirv(std::string_view glsl_source, std::string_view name, shader_kind kind)
{
  ensure_glslang();

  const EShLanguage stage = to_glslang(kind);
  glslang::TShader shader(stage);
  const char *strings[] = { glsl_source.data() };
  const int lengths[] = { static_cast<int>(glsl_source.size()) };
  shader.setStringsWithLengths(strings, lengths, 1);
  shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
  shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);
  shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_5);

  const TBuiltInResource &resources = *GetDefaultResources();
  EShMessages messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
  if (!shader.parse(&resources, 450, false, messages)) {
    throw std::runtime_error(std::string("glslang parse failed for ") + std::string(name) + ":\n" + shader.getInfoLog());
  }

  glslang::TProgram program;
  program.addShader(&shader);
  if (!program.link(messages)) {
    throw std::runtime_error(std::string("glslang link failed for ") + std::string(name) + ":\n" + program.getInfoLog());
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

} // namespace vkexec
