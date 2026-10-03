cmake_minimum_required(VERSION 3.29)

project(
  vkexec
  VERSION 0.2.0
  DESCRIPTION "stdexec Vulkan compute backend with optional extensions and graphics helpers"
  HOMEPAGE_URL "https://github.com/janagor/vkexec"
  LANGUAGES CXX C)

include(cmake/PreventInSourceBuilds.cmake)
include(GNUInstallDirs)
include(ProjectOptions.cmake)

vkexec_setup_options()

vkexec_global_options()
include(Dependencies.cmake)
vkexec_setup_dependencies()

vkexec_local_options()

set(GIT_SHA
    "Unknown"
    CACHE STRING "SHA this build was generated from")
string(
  SUBSTRING "${GIT_SHA}"
            0
            8
            GIT_SHORT_SHA)

add_library(vkexec::vkexec_options ALIAS vkexec_options)
add_library(vkexec::vkexec_warnings ALIAS vkexec_warnings)

# configure files based on CMake configuration options
add_subdirectory(configured_files)

# Adding the library source:
add_subdirectory(src)

# Public headers and installed template implementation-support headers are owned
# by exported targets. Source-only detail headers are deliberately absent from
# these sets and can never enter an install package.
file(
  GLOB_RECURSE
  VKEXEC_CORE_HEADERS
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/vkexec/*.hpp")
file(
  GLOB_RECURSE
  VKEXEC_VMA_HEADERS
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/vkexec_vma/*.hpp")
file(
  GLOB_RECURSE
  VKEXEC_TOOLS_HEADERS
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/vkexec_tools/*.hpp")
file(
  GLOB_RECURSE
  VKEXEC_FEATURE_HEADERS
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/vkexec_features/*.hpp")
file(
  GLOB_RECURSE
  VKEXEC_GRAPHICS_HEADERS
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/vkexec_graphics/*.hpp")
file(
  GLOB_RECURSE
  VKEXEC_DESCRIPTOR_HEAP_HEADERS
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/vkexec_extensions/common/*.hpp"
  "${PROJECT_SOURCE_DIR}/include/vkexec_extensions/descriptor_heap/*.hpp")
list(
  APPEND
  VKEXEC_DESCRIPTOR_HEAP_HEADERS
  "${PROJECT_SOURCE_DIR}/include/vkexec_extensions/descriptor_heap.hpp"
  "${PROJECT_SOURCE_DIR}/include/vkexec_extensions/extension.hpp")
file(
  GLOB_RECURSE
  VKEXEC_DYNAMIC_RENDERING_HEADERS
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/vkexec_extensions/dynamic_rendering/*.hpp")
list(APPEND VKEXEC_DYNAMIC_RENDERING_HEADERS "${PROJECT_SOURCE_DIR}/include/vkexec_extensions/dynamic_rendering.hpp")
file(
  GLOB_RECURSE
  VKEXEC_TIMELINE_HEADERS
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/vkexec_extensions/timeline_semaphore/*.hpp")
list(APPEND VKEXEC_TIMELINE_HEADERS "${PROJECT_SOURCE_DIR}/include/vkexec_extensions/timeline_semaphore.hpp")

target_sources(
  vkexec
  PUBLIC FILE_SET
         public_headers
         TYPE
         HEADERS
         BASE_DIRS
         "${PROJECT_SOURCE_DIR}/include"
         FILES
         ${VKEXEC_CORE_HEADERS})
if(vkexec_BUILD_VMA)
  target_sources(
    vkexec_vma
    PUBLIC FILE_SET
           public_headers
           TYPE
           HEADERS
           BASE_DIRS
           "${PROJECT_SOURCE_DIR}/include"
           FILES
           ${VKEXEC_VMA_HEADERS})
endif()
if(vkexec_BUILD_TOOLS)
  target_sources(
    vkexec_tools
    PUBLIC FILE_SET
           public_headers
           TYPE
           HEADERS
           BASE_DIRS
           "${PROJECT_SOURCE_DIR}/include"
           FILES
           ${VKEXEC_TOOLS_HEADERS})
endif()
target_sources(
  vkexec_features
  PUBLIC FILE_SET
         public_headers
         TYPE
         HEADERS
         BASE_DIRS
         "${PROJECT_SOURCE_DIR}/include"
         FILES
         ${VKEXEC_FEATURE_HEADERS})
target_sources(
  vkexec_graphics
  PUBLIC FILE_SET
         public_headers
         TYPE
         HEADERS
         BASE_DIRS
         "${PROJECT_SOURCE_DIR}/include"
         FILES
         ${VKEXEC_GRAPHICS_HEADERS})
target_sources(
  vkexec_ext_descriptor_heap
  PUBLIC FILE_SET
         public_headers
         TYPE
         HEADERS
         BASE_DIRS
         "${PROJECT_SOURCE_DIR}/include"
         FILES
         ${VKEXEC_DESCRIPTOR_HEAP_HEADERS})
target_sources(
  vkexec_ext_dynamic_rendering
  PUBLIC FILE_SET
         public_headers
         TYPE
         HEADERS
         BASE_DIRS
         "${PROJECT_SOURCE_DIR}/include"
         FILES
         ${VKEXEC_DYNAMIC_RENDERING_HEADERS})
target_sources(
  vkexec_ext_timeline_semaphore
  PUBLIC FILE_SET
         public_headers
         TYPE
         HEADERS
         BASE_DIRS
         "${PROJECT_SOURCE_DIR}/include"
         FILES
         ${VKEXEC_TIMELINE_HEADERS})

if(vkexec_BUILD_EXAMPLES AND vkexec_BUILD_VMA)
  add_subdirectory(examples)
elseif(vkexec_BUILD_EXAMPLES)
  message(STATUS "Examples require the optional VMA backend and will not be built")
endif()

if(PROJECT_IS_TOP_LEVEL)
  include(cmake/Format.cmake)
  vkexec_enable_formatting()

  # Adding the tests:
  include(CTest)

  if(vkexec_ENABLE_INSTALL_CONSUMER_TEST)
    get_property(VKEXEC_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    add_custom_target(
      vkexec_install_consumer
      COMMAND
        "${CMAKE_COMMAND}" "-DVKEXEC_SOURCE_DIR=${vkexec_SOURCE_DIR}" "-DVKEXEC_BINARY_DIR=${vkexec_BINARY_DIR}"
        "-DCONSUMER_SOURCE_DIR=${vkexec_SOURCE_DIR}/test/fixtures/install_consumer" "-DVKEXEC_CONFIG=$<CONFIG>"
        "-DVKEXEC_EXPECT_VMA=${vkexec_BUILD_VMA}" "-DVKEXEC_EXPECT_TOOLS=${vkexec_BUILD_TOOLS}"
        "-DVKEXEC_EXPECT_SHARED=${BUILD_SHARED_LIBS}" "-DVKEXEC_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        "-DVKEXEC_CXX_COMPILER=${CMAKE_CXX_COMPILER}" "-DVKEXEC_GENERATOR=${CMAKE_GENERATOR}"
        "-DVKEXEC_GENERATOR_IS_MULTI_CONFIG=${VKEXEC_GENERATOR_IS_MULTI_CONFIG}"
        "-DVKEXEC_GENERATOR_INSTANCE=${CMAKE_GENERATOR_INSTANCE}"
        "-DVKEXEC_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
        "-DVKEXEC_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}" -P
        "${vkexec_SOURCE_DIR}/cmake/TestInstallConsumer.cmake"
      VERBATIM)
  endif()

  if(BUILD_TESTING)
    add_subdirectory(test)
    add_test(NAME vkexec.documentation_consistency
             COMMAND "${CMAKE_COMMAND}" "-DVKEXEC_SOURCE_DIR=${PROJECT_SOURCE_DIR}" -P
                     "${PROJECT_SOURCE_DIR}/cmake/CheckDocumentation.cmake")
    add_test(NAME vkexec.cxx_standard_policy COMMAND "${CMAKE_COMMAND}" "-DVKEXEC_SOURCE_DIR=${PROJECT_SOURCE_DIR}" -P
                                                     "${PROJECT_SOURCE_DIR}/cmake/CheckCxxStandardPolicy.cmake")
    add_test(NAME vkexec.stdexec_public_api_boundary
             COMMAND "${CMAKE_COMMAND}" "-DVKEXEC_SOURCE_DIR=${PROJECT_SOURCE_DIR}" -P
                     "${PROJECT_SOURCE_DIR}/cmake/CheckStdexecPrivateApi.cmake")
    add_test(NAME vkexec.vma_module_boundary COMMAND "${CMAKE_COMMAND}" "-DVKEXEC_SOURCE_DIR=${PROJECT_SOURCE_DIR}" -P
                                                     "${PROJECT_SOURCE_DIR}/cmake/CheckVmaBoundary.cmake")
    add_test(NAME vkexec.vk_bootstrap_boundary COMMAND "${CMAKE_COMMAND}" "-DVKEXEC_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
                                                       -P "${PROJECT_SOURCE_DIR}/cmake/CheckVkBootstrapBoundary.cmake")
  endif()

  if(MSVC AND vkexec_ENABLE_SANITIZER_ADDRESS)
    get_all_installable_targets(all_targets)
    if(all_targets)
      set_target_properties(${all_targets} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=$(VC_ExecutablePath_x64);%PATH%")
    endif()
  endif()

  if(TARGET triangle)
    set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT triangle)
  endif()

  if(NOT CMAKE_SKIP_INSTALL_RULES)
    include(CMakePackageConfigHelpers)

    set(VKEXEC_INSTALL_TARGETS
        vkexec
        vkexec_features
        vkexec_ext_descriptor_heap
        vkexec_ext_dynamic_rendering
        vkexec_ext_timeline_semaphore
        vkexec_graphics)
    if(vkexec_BUILD_VMA)
      list(APPEND VKEXEC_INSTALL_TARGETS vkexec_vma)
    endif()
    if(vkexec_BUILD_TOOLS)
      list(APPEND VKEXEC_INSTALL_TARGETS vkexec_tools)
    endif()
    # The vkexec export contains only vkexec-owned public targets. Third-party
    # dependencies are restored by vkexecConfig.cmake through their own packages.
    set(VKEXEC_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/vkexec")
    set(VKEXEC_PACKAGE_NEEDS_VK_BOOTSTRAP OFF)
    set(VKEXEC_PACKAGE_NEEDS_GLSLANG OFF)
    if(NOT BUILD_SHARED_LIBS)
      set(VKEXEC_PACKAGE_NEEDS_VK_BOOTSTRAP ON)
      if(vkexec_BUILD_TOOLS)
        set(VKEXEC_PACKAGE_NEEDS_GLSLANG ON)
      endif()
    endif()
    install(
      TARGETS ${VKEXEC_INSTALL_TARGETS}
      EXPORT vkexecTargets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
              FILE_SET public_headers
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(
      EXPORT vkexecTargets
      FILE vkexecTargets.cmake
      NAMESPACE vkexec::
      DESTINATION "${VKEXEC_INSTALL_CMAKEDIR}")
    configure_package_config_file(
      "${PROJECT_SOURCE_DIR}/cmake/vkexecConfig.cmake.in" "${PROJECT_BINARY_DIR}/vkexecConfig.cmake"
      INSTALL_DESTINATION "${VKEXEC_INSTALL_CMAKEDIR}")
    write_basic_package_version_file(
      "${PROJECT_BINARY_DIR}/vkexecConfigVersion.cmake"
      VERSION "${PROJECT_VERSION}"
      COMPATIBILITY SameMajorVersion)
    install(FILES "${PROJECT_BINARY_DIR}/vkexecConfig.cmake" "${PROJECT_BINARY_DIR}/vkexecConfigVersion.cmake"
            DESTINATION "${VKEXEC_INSTALL_CMAKEDIR}")

    install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/${PROJECT_NAME}")

    # Experience shows that explicit package naming can help make it easier to sort
    # out potential ABI related issues before they start, while helping you
    # track a build to a specific GIT SHA
    set(CPACK_PACKAGE_FILE_NAME
        "${CMAKE_PROJECT_NAME}-${CMAKE_PROJECT_VERSION}-${GIT_SHORT_SHA}-${CMAKE_SYSTEM_NAME}-${CMAKE_BUILD_TYPE}-${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
    )
    set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")

    include(CPack)
  endif()
endif()
