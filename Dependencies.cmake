include(cmake/CPM.cmake)

# Done as a function so that updates to variables like
# CMAKE_CXX_FLAGS don't propagate out to other
# targets
function(vkexec_setup_dependencies)

  # For each dependency, see if it's
  # already been provided to us by a parent project

  if(NOT TARGET fmtlib::fmtlib)
    cpmaddpackage(
      NAME
      fmt
      GITHUB_REPOSITORY
      "fmtlib/fmt"
      GIT_TAG
      "12.1.0"
      SYSTEM
      YES)
  endif()

  if(NOT TARGET spdlog::spdlog)
    cpmaddpackage(
      NAME
      spdlog
      VERSION
      1.17.0
      GITHUB_REPOSITORY
      "gabime/spdlog"
      SYSTEM
      YES
      OPTIONS
      "SPDLOG_FMT_EXTERNAL ON")
  endif()

  if(NOT TARGET Catch2::Catch2WithMain)
    cpmaddpackage(
      NAME
      Catch2
      VERSION
      3.12.0
      GITHUB_REPOSITORY
      "catchorg/Catch2"
      SYSTEM
      YES)
  endif()

  if(NOT TARGET CLI11::CLI11)
    cpmaddpackage(
      NAME
      CLI11
      VERSION
      2.6.1
      GITHUB_REPOSITORY
      "CLIUtils/CLI11"
      SYSTEM
      YES)
  endif()

  if(NOT TARGET ftxui::screen)
    cpmaddpackage(
      NAME
      FTXUI
      VERSION
      6.1.9
      GITHUB_REPOSITORY
      "ArthurSonzogni/FTXUI"
      SYSTEM
      YES)
  endif()

  if(NOT TARGET tools::tools)
    cpmaddpackage(
      NAME
      tools
      GITHUB_REPOSITORY
      "lefticus/tools"
      GIT_TAG
      "main")
  endif()

  if(NOT TARGET glslang::glslang)
    cpmaddpackage(
      NAME
      glslang
      GITHUB_REPOSITORY
      "KhronosGroup/glslang"
      GIT_TAG
      "15.4.0"
      SYSTEM
      YES
      OPTIONS
      "ENABLE_OPT OFF"
      "ENABLE_HLSL OFF"
      "ENABLE_GLSLANG_BINARIES OFF"
      "ENABLE_SPVREMAPPER OFF"
      "GLSLANG_ENABLE_INSTALL OFF"
      "GLSLANG_TESTS OFF"
      "BUILD_EXTERNAL OFF"
      "ENABLE_PCH OFF")
  endif()

  if(NOT TARGET STDEXEC::stdexec)
    cpmaddpackage(
      NAME
      stdexec
      GITHUB_REPOSITORY
      "NVIDIA/stdexec"
      GIT_TAG
      "main"
      SYSTEM
      YES
      OPTIONS
      "STDEXEC_BUILD_EXAMPLES OFF"
      "STDEXEC_BUILD_TESTS OFF"
      "STDEXEC_ENABLE_CUDA OFF"
      "STDEXEC_ENABLE_IO_URING OFF")
  endif()

  if(NOT TARGET glfw)
    cpmaddpackage(
      NAME
      glfw
      GITHUB_REPOSITORY
      "glfw/glfw"
      GIT_TAG
      "3.4"
      SYSTEM
      YES
      OPTIONS
      "GLFW_BUILD_EXAMPLES OFF"
      "GLFW_BUILD_TESTS OFF"
      "GLFW_BUILD_DOCS OFF"
      "GLFW_INSTALL OFF"
      "GLFW_BUILD_WAYLAND ON"
      "GLFW_BUILD_X11 ON")
  endif()

  find_package(Vulkan REQUIRED)

  if(NOT TARGET vk-bootstrap::vk-bootstrap)
    cpmaddpackage(
      NAME
      vk-bootstrap
      GITHUB_REPOSITORY
      "charles-lunarg/vk-bootstrap"
      GIT_TAG
      "v1.4.352"
      SYSTEM
      YES)
  endif()

  if(NOT TARGET GPUOpen::VulkanMemoryAllocator)
    cpmaddpackage(
      NAME
      VulkanMemoryAllocator
      GITHUB_REPOSITORY
      "GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator"
      GIT_TAG
      "v3.4.0"
      SYSTEM
      YES)
  endif()

endfunction()
