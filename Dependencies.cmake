include(cmake/CPM.cmake)

# Done as a function so that updates to variables like
# CMAKE_CXX_FLAGS don't propagate out to other
# targets
function(vkexec_setup_dependencies)

  # For each dependency, see if it's
  # already been provided to us by a parent project

  # Pin headers before find_package(Vulkan): FindVulkan creates Vulkan::Headers
  # from the system/SDK install when present, which can be older than the
  # vk-bootstrap tag (e.g. Ubuntu libvulkan-dev) and then skips CPM.
  # Use DOWNLOAD_ONLY + IMPORTED so export still treats Headers like FindVulkan
  # (consumers resolve it via find_dependency(Vulkan)).
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
    set_target_properties(Vulkan::Headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${VulkanHeaders_SOURCE_DIR}/include")
  endif()

  if(NOT TARGET Vulkan::Vulkan)
    find_package(Vulkan QUIET)
    if(NOT Vulkan_FOUND)
      find_package(PkgConfig QUIET)
      if(PkgConfig_FOUND)
        pkg_check_modules(Vulkan IMPORTED_TARGET vulkan)
        if(TARGET PkgConfig::Vulkan)
          add_library(Vulkan::Vulkan ALIAS PkgConfig::Vulkan)
        endif()
      endif()
    endif()
    if(NOT TARGET Vulkan::Vulkan)
      message(FATAL_ERROR "Vulkan loader not found (Vulkan::Vulkan). Install the Vulkan SDK "
                          "(set VULKAN_SDK) or a system package such as libvulkan-dev.")
    endif()
  endif()

  # Prefer the pinned Vulkan::Headers over any system/SDK include dirs that
  # FindVulkan / pkg-config attach to the loader target.
  if(TARGET Vulkan::Vulkan)
    get_target_property(_vkexec_vulkan_aliased Vulkan::Vulkan ALIASED_TARGET)
    if(_vkexec_vulkan_aliased)
      set_property(TARGET ${_vkexec_vulkan_aliased} PROPERTY INTERFACE_INCLUDE_DIRECTORIES "")
    else()
      set_property(TARGET Vulkan::Vulkan PROPERTY INTERFACE_INCLUDE_DIRECTORIES "")
    endif()
    unset(_vkexec_vulkan_aliased)
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

  if(NOT TARGET Boost::system)
    cpmaddpackage(
      NAME
      Boost
      VERSION
      1.86.0
      URL
      https://github.com/boostorg/boost/releases/download/boost-1.86.0/boost-1.86.0-cmake.tar.xz
      URL_HASH
      SHA256=2c5ec5edcdff47ff55e27ed9560b0a0b94b07bd07ed9928b476150e16b0efc57
      SYSTEM
      YES
      OPTIONS
      "BOOST_ENABLE_CMAKE ON"
      "BOOST_SKIP_INSTALL_RULES ON"
      "BUILD_SHARED_LIBS OFF"
      "BOOST_INCLUDE_LIBRARIES system")
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
      "GLSLANG_ENABLE_INSTALL OFF"
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
        target_include_directories(
          ${_vkexec_glslang_target}
          INTERFACE $<BUILD_INTERFACE:${_vkexec_glslang_compat_include}>)
      endif()
    endforeach()
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
    target_compile_definitions(tinygltf PRIVATE TINYGLTF3_ENABLE_FS)
    set_target_properties(
      tinygltf
      PROPERTIES C_STANDARD 11
                 C_STANDARD_REQUIRED ON
                 C_CLANG_TIDY ""
                 C_CPPCHECK "")
    set_source_files_properties(${tinygltf_SOURCE_DIR}/tiny_gltf_v3.c PROPERTIES SKIP_LINTING ON COMPILE_OPTIONS
                                                                                                 "-Wno-everything")
  endif()

endfunction()
