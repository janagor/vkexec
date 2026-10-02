include(cmake/CPM.cmake)

set(VKEXEC_STDEXEC_GIT_TAG
    "ead186b1d8db3ebe37a946ff84a6ce08bf795153"
    CACHE STRING "NVIDIA/stdexec revision used by vkexec")

# Done as a function so that updates to variables like
# CMAKE_CXX_FLAGS don't propagate out to other
# targets
function(vkexec_setup_dependencies)
  set(_vkexec_install_dependencies OFF)
  if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_SKIP_INSTALL_RULES)
    set(_vkexec_install_dependencies ON)
  endif()

  # For each dependency, see if it's
  # already been provided to us by a parent project

  # Respect parent targets, then prefer an installed Vulkan SDK or system package.
  if(NOT TARGET Vulkan::Headers OR NOT TARGET Vulkan::Vulkan)
    find_package(Vulkan QUIET)
  endif()

  if(NOT TARGET Vulkan::Headers)
    find_package(VulkanHeaders CONFIG QUIET)
  endif()

  if(NOT TARGET Vulkan::Vulkan)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(Vulkan IMPORTED_TARGET vulkan)
      if(TARGET PkgConfig::Vulkan)
        add_library(Vulkan::Vulkan ALIAS PkgConfig::Vulkan)
      endif()
    endif()
    if(NOT TARGET Vulkan::Vulkan)
      message(FATAL_ERROR "Vulkan loader not found (Vulkan::Vulkan). Install the Vulkan SDK "
                          "(set VULKAN_SDK) or a system package such as libvulkan-dev.")
    endif()
  endif()

  # Download headers only when neither the parent nor an installed package provided them.
  # Keep the fallback behind the standard Vulkan::Headers target.
  if(NOT TARGET Vulkan::Headers)
    cpmaddpackage(
      NAME
      VulkanHeaders
      GITHUB_REPOSITORY
      KhronosGroup/Vulkan-Headers
      GIT_TAG
      "v1.4.352"
      DOWNLOAD_ONLY
      YES
      SYSTEM
      YES)
    add_library(Vulkan::Headers INTERFACE IMPORTED)
    set_target_properties(Vulkan::Headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                                     "${VulkanHeaders_SOURCE_DIR}/include")
  endif()

  # Compose existing usage requirements without changing third-party targets.
  add_library(vkexec_vulkan INTERFACE)
  target_link_libraries(vkexec_vulkan INTERFACE Vulkan::Headers "$<LINK_ONLY:Vulkan::Vulkan>")

  if((NOT DEFINED BUILD_TESTING OR BUILD_TESTING) AND NOT TARGET Catch2::Catch2WithMain)
    cpmaddpackage(
      NAME
      Catch2
      VERSION
      3.12.0
      GITHUB_REPOSITORY
      "catchorg/Catch2"
      SYSTEM
      YES)
    target_compile_features(Catch2 PRIVATE cxx_std_17)
  endif()

  if(vkexec_BUILD_TOOLS AND NOT TARGET glslang::glslang)
    find_package(glslang CONFIG QUIET)
  endif()
  if(vkexec_BUILD_TOOLS AND NOT TARGET glslang::glslang)
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
      "GLSLANG_ENABLE_INSTALL ${_vkexec_install_dependencies}"
      "GLSLANG_TESTS OFF"
      "BUILD_EXTERNAL OFF"
      "ENABLE_PCH OFF")

    # glslang BUILD_INTERFACE exposes the repo root (SPIRV/...), while the
    # install layout and public includes use glslang/SPIRV/... Mirror that
    # layout in the build tree so #include <glslang/SPIRV/...> works via CPM.
    # See KhronosGroup/glslang#4185 / #4298.
    set(_vkexec_glslang_compat_include "${CMAKE_BINARY_DIR}/_deps/glslang-compat-include")
    file(MAKE_DIRECTORY "${_vkexec_glslang_compat_include}/glslang")
    if(NOT EXISTS "${_vkexec_glslang_compat_include}/glslang/SPIRV")
      file(
        CREATE_LINK
        "${glslang_SOURCE_DIR}/SPIRV"
        "${_vkexec_glslang_compat_include}/glslang/SPIRV"
        SYMBOLIC
        COPY_ON_ERROR)
    endif()
    foreach(_vkexec_glslang_target IN ITEMS glslang SPIRV glslang-default-resource-limits)
      if(TARGET ${_vkexec_glslang_target})
        target_include_directories(${_vkexec_glslang_target}
                                   INTERFACE $<BUILD_INTERFACE:${_vkexec_glslang_compat_include}>)
        if(MSVC)
          # glslang still uses strncpy; MSVC C4996 becomes an error under /WX-capable CI.
          target_compile_definitions(${_vkexec_glslang_target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        endif()
      endif()
    endforeach()
  endif()

  if(NOT TARGET STDEXEC::stdexec)
    find_package(stdexec CONFIG QUIET)
  endif()
  if(NOT TARGET STDEXEC::stdexec)
    cpmaddpackage(
      NAME
      stdexec
      GITHUB_REPOSITORY
      "NVIDIA/stdexec"
      GIT_TAG
      "${VKEXEC_STDEXEC_GIT_TAG}"
      SYSTEM
      YES
      OPTIONS
      "STDEXEC_BUILD_EXAMPLES OFF"
      "STDEXEC_BUILD_TESTS OFF"
      "STDEXEC_INSTALL ${_vkexec_install_dependencies}"
      "STDEXEC_ENABLE_CUDA OFF"
      "STDEXEC_ENABLE_IO_URING OFF")
  endif()

  if(vkexec_BUILD_EXAMPLES AND NOT TARGET glfw)
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

  if(NOT TARGET vk-bootstrap::vk-bootstrap)
    find_package(vk-bootstrap CONFIG QUIET)
  endif()
  if(NOT TARGET vk-bootstrap::vk-bootstrap)
    cpmaddpackage(
      NAME
      vk-bootstrap
      GITHUB_REPOSITORY
      "charles-lunarg/vk-bootstrap"
      GIT_TAG
      "v1.4.352"
      SYSTEM
      YES
      OPTIONS
      "VK_BOOTSTRAP_INSTALL ${_vkexec_install_dependencies}")
  endif()

  if(vkexec_BUILD_VMA AND NOT TARGET GPUOpen::VulkanMemoryAllocator)
    find_package(VulkanMemoryAllocator CONFIG QUIET)
  endif()
  if(vkexec_BUILD_VMA AND NOT TARGET GPUOpen::VulkanMemoryAllocator)
    cpmaddpackage(
      NAME
      VulkanMemoryAllocator
      GITHUB_REPOSITORY
      "GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator"
      GIT_TAG
      "v3.4.0"
      SYSTEM
      YES
      OPTIONS
      "VMA_ENABLE_INSTALL ${_vkexec_install_dependencies}")
  endif()

  if(vkexec_BUILD_EXAMPLES AND NOT TARGET tinygltf::tinygltf)
    cpmaddpackage(
      NAME
      tinygltf
      GITHUB_REPOSITORY
      "syoyo/tinygltf"
      GIT_TAG
      "v3.0.1"
      DOWNLOAD_ONLY
      YES
      SYSTEM
      YES)

    add_library(tinygltf STATIC ${tinygltf_SOURCE_DIR}/tiny_gltf_v3.c)
    add_library(tinygltf::tinygltf ALIAS tinygltf)

    target_include_directories(tinygltf SYSTEM PUBLIC ${tinygltf_SOURCE_DIR})
    target_compile_definitions(tinygltf PRIVATE TINYGLTF3_ENABLE_FS $<$<C_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>)
    target_compile_options(
      tinygltf PRIVATE $<$<C_COMPILER_ID:MSVC>:/w> $<$<C_COMPILER_ID:Clang,AppleClang>:-Wno-everything>
                       $<$<C_COMPILER_ID:GNU>:-w>)
    set_target_properties(
      tinygltf
      PROPERTIES C_STANDARD 11
                 C_STANDARD_REQUIRED ON
                 C_CLANG_TIDY ""
                 C_CPPCHECK "")
    set_source_files_properties(${tinygltf_SOURCE_DIR}/tiny_gltf_v3.c PROPERTIES SKIP_LINTING ON)
  endif()

endfunction()
