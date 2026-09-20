cmake_minimum_required(VERSION 3.29)

# Disable modules support, this is broken with clang-tidy at the moment
cmake_policy(SET CMP0155 OLD)

# This template attempts to be "fetch_content"-able
# so that it works well with tools like CPM or other
# manual dependency management

# Only set the cxx_standard if it is not set by someone else
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

# strongly encouraged to enable this globally to avoid conflicts between
# -Wpedantic being enabled and -std=c++20 and -std=gnu++20 for example
# when compiling with PCH enabled
set(CMAKE_CXX_EXTENSIONS OFF)

# Set the project name and language
project(
  vkexec
  VERSION 0.2.0
  DESCRIPTION "stdexec Vulkan compute backend with optional extensions and graphics helpers"
  HOMEPAGE_URL "https://github.com/janagor/vkexec"
  LANGUAGES CXX C)

include(cmake/PreventInSourceBuilds.cmake)
include(ProjectOptions.cmake)

vkexec_setup_options()

vkexec_global_options()
include(Dependencies.cmake)
vkexec_setup_dependencies()

vkexec_local_options()

# don't know if this should be set globally from here or not...
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

set(GIT_SHA
    "Unknown"
    CACHE STRING "SHA this build was generated from")
string(
  SUBSTRING "${GIT_SHA}"
            0
            8
            GIT_SHORT_SHA)

target_compile_features(vkexec_options INTERFACE cxx_std_${CMAKE_CXX_STANDARD})

add_library(vkexec::vkexec_options ALIAS vkexec_options)
add_library(vkexec::vkexec_warnings ALIAS vkexec_warnings)

#add_library(vkexec::vkexec_options INTERFACE IMPORTED)
#add_library(vkexec::vkexec_warnings INTERFACE IMPORTED)

# configure files based on CMake configuration options
add_subdirectory(configured_files)

# Adding the library source:
add_subdirectory(src)

# Public headers are owned by exported targets. Source-only detail headers are
# deliberately absent from these sets and can never enter an install package.
file(GLOB VKEXEC_CORE_HEADERS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/vkexec/*.hpp")
file(GLOB_RECURSE VKEXEC_FEATURE_HEADERS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/vkexec_features/*.hpp")
file(GLOB_RECURSE VKEXEC_GRAPHICS_HEADERS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/vkexec_graphics/*.hpp")
file(GLOB_RECURSE VKEXEC_DESCRIPTOR_HEAP_HEADERS CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/vkexec_extensions/common/*.hpp"
  "${PROJECT_SOURCE_DIR}/include/vkexec_extensions/descriptor_heap/*.hpp")
list(APPEND VKEXEC_DESCRIPTOR_HEAP_HEADERS
  "${PROJECT_SOURCE_DIR}/include/vkexec_extensions/descriptor_heap.hpp"
  "${PROJECT_SOURCE_DIR}/include/vkexec_extensions/extension.hpp")
file(GLOB_RECURSE VKEXEC_DYNAMIC_RENDERING_HEADERS CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/vkexec_extensions/dynamic_rendering/*.hpp")
list(APPEND VKEXEC_DYNAMIC_RENDERING_HEADERS
  "${PROJECT_SOURCE_DIR}/include/vkexec_extensions/dynamic_rendering.hpp")
file(GLOB_RECURSE VKEXEC_TIMELINE_HEADERS CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/vkexec_extensions/timeline_semaphore/*.hpp")
list(APPEND VKEXEC_TIMELINE_HEADERS "${PROJECT_SOURCE_DIR}/include/vkexec_extensions/timeline_semaphore.hpp")

target_sources(vkexec PUBLIC FILE_SET public_headers TYPE HEADERS BASE_DIRS "${PROJECT_SOURCE_DIR}/include"
                            FILES ${VKEXEC_CORE_HEADERS})
target_sources(vkexec_features PUBLIC FILE_SET public_headers TYPE HEADERS BASE_DIRS "${PROJECT_SOURCE_DIR}/include"
                                     FILES ${VKEXEC_FEATURE_HEADERS})
target_sources(vkexec_graphics PUBLIC FILE_SET public_headers TYPE HEADERS BASE_DIRS "${PROJECT_SOURCE_DIR}/include"
                                     FILES ${VKEXEC_GRAPHICS_HEADERS})
target_sources(vkexec_ext_descriptor_heap PUBLIC FILE_SET public_headers TYPE HEADERS
  BASE_DIRS "${PROJECT_SOURCE_DIR}/include" FILES ${VKEXEC_DESCRIPTOR_HEAP_HEADERS})
target_sources(vkexec_ext_dynamic_rendering PUBLIC FILE_SET public_headers TYPE HEADERS
  BASE_DIRS "${PROJECT_SOURCE_DIR}/include" FILES ${VKEXEC_DYNAMIC_RENDERING_HEADERS})
target_sources(vkexec_ext_timeline_semaphore PUBLIC FILE_SET public_headers TYPE HEADERS
  BASE_DIRS "${PROJECT_SOURCE_DIR}/include" FILES ${VKEXEC_TIMELINE_HEADERS})

if(vkexec_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Don't even look at tests if we're not top level
if(NOT PROJECT_IS_TOP_LEVEL)
  return()
endif()

include(cmake/Format.cmake)
vkexec_enable_formatting()

# Adding the tests:
include(CTest)

if(BUILD_TESTING)
  message(AUTHOR_WARNING "Building Tests. Be sure to check out test/constexpr_tests.cpp for constexpr testing")
  add_subdirectory(test)
endif()

# If MSVC is being used, and ASAN is enabled, we need to set the debugger environment
# so that it behaves well with MSVC's debugger, and we can run the target from visual studio
if(MSVC)
  get_all_installable_targets(all_targets)
  message("all_targets=${all_targets}")
  set_target_properties(${all_targets} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=$(VC_ExecutablePath_x64);%PATH%")
endif()

# set the startup project for the "play" button in MSVC
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT vkexec_triangle_example)

if(CMAKE_SKIP_INSTALL_RULES)
  return()
endif()

include(cmake/PackageProject.cmake)

# Interface option/warning targets remain package-exportable.
vkexec_package_project(TARGETS vkexec_options vkexec_warnings)

# Static libraries pull CPM targets that are not themselves exported. Install the
# archives and public headers so consumers can still stage a prefix; prefer
# add_subdirectory / CPM for full target propagation until deps are packaged.
include(GNUInstallDirs)
install(
  TARGETS vkexec
          vkexec_features
          vkexec_ext_descriptor_heap
          vkexec_ext_dynamic_rendering
          vkexec_ext_timeline_semaphore
          vkexec_graphics
  FILE_SET public_headers DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

# Experience shows that explicit package naming can help make it easier to sort
# out potential ABI related issues before they start, while helping you
# track a build to a specific GIT SHA
set(CPACK_PACKAGE_FILE_NAME
    "${CMAKE_PROJECT_NAME}-${CMAKE_PROJECT_VERSION}-${GIT_SHORT_SHA}-${CMAKE_SYSTEM_NAME}-${CMAKE_BUILD_TYPE}-${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
)

include(CPack)
