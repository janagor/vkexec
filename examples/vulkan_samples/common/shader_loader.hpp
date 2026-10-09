#ifndef VKEXEC_VULKAN_SAMPLES_SHADER_LOADER_HPP
#define VKEXEC_VULKAN_SAMPLES_SHADER_LOADER_HPP

#include "sync_wait_helpers.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace vkexec::examples {

[[nodiscard]] inline auto load_spirv(std::filesystem::path const &path) -> std::vector<std::uint32_t>
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) { fail_check("failed to open sample shader"); }

  auto const bytes = file.tellg();
  if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(std::uint32_t)) != 0) {
    fail_check("sample shader is not valid SPIR-V data");
  }

  std::vector<std::uint32_t> words(static_cast<std::size_t>(bytes) / sizeof(std::uint32_t));
  file.seekg(0);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  file.read(reinterpret_cast<char *>(words.data()), static_cast<std::streamsize>(bytes));
  if (!file) { fail_check("failed to read sample shader"); }
  return words;
}

}// namespace vkexec::examples

#endif// VKEXEC_VULKAN_SAMPLES_SHADER_LOADER_HPP
