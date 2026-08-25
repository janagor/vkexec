cmake_minimum_required(VERSION 3.15...3.23)

project(
  CmakeConfigPackageTests
  VERSION ${PROJECT_VERSION}
  LANGUAGES CXX)

# Workaround clang>=22 flagging __COUNTER__ as c2y-extension
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 22)
  add_definitions(-DCATCH_CONFIG_NO_COUNTER)
endif()

# ---- Test as standalone project the exported config package ----

if(PROJECT_IS_TOP_LEVEL OR TEST_INSTALLED_VERSION)
  enable_testing()

  find_package(vkexec CONFIG REQUIRED) # for vkexec, project_options, ...

  if(NOT TARGET vkexec_options)
    message(FATAL_ERROR "Required config package not found!")
    return() # be strictly paranoid for Template Janitor github action! CK
  endif()
endif()

# ---- Dependencies ----

include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)

# CMake-script regression tests for cmake/Sanitizers.cmake (issue #141).
add_test(NAME cmake.sanitizers.apple_excludes_leak
         COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_sanitizers_apple_excludes_leak.cmake)
add_test(NAME cmake.sanitizers.non_apple_includes_leak
         COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_sanitizers_non_apple_includes_leak.cmake)

add_executable(
  tests
  tests.cpp
  pass_detail_tests.cpp
  dispatch_tests.cpp
  edsl_emit_tests.cpp
  edsl_types_tests.cpp
  edsl_spirv_tests.cpp
  error_tests.cpp
  push_constant_tests.cpp
  bulk_tests.cpp
  buffer_tests.cpp
  gpu_buffer_tests.cpp
  timeline_semaphore_tests.cpp
  frame_ring_tests.cpp
  push_data_tests.cpp
  descriptor_heap_tests.cpp
  descriptor_heap_pipeline_tests.cpp
  image_tests.cpp
  image_barrier_tests.cpp
  rendering_tests.cpp
  sampler_tests.cpp
  scheduler_tests.cpp
  graphics_tests.cpp
  draw_stop_tests.cpp
  vulkan_requirements_tests.cpp
  gpu_tests.cpp)
target_link_libraries(
  tests
  PRIVATE vkexec::vkexec_warnings
          vkexec::vkexec_options
          vkexec::vkexec
          vkexec::vkexec_edsl
          vkexec::vkexec_graphics
          Catch2::Catch2WithMain)

if(EXISTS "${CMAKE_SOURCE_DIR}/src/vkexec_edsl/glsl_emit.hpp")
  target_include_directories(tests PRIVATE "${CMAKE_SOURCE_DIR}/src")
endif()

vkexec_configure_linker(tests)

if(WIN32 AND BUILD_SHARED_LIBS)
  add_custom_command(
    TARGET tests
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:tests> $<TARGET_FILE_DIR:tests>
    COMMAND_EXPAND_LISTS)
endif()

# automatically discover tests that are defined in catch based test files you can modify the unittests. Set TEST_PREFIX
# to whatever you want, or use different for different binaries
catch_discover_tests(
  tests
  TEST_PREFIX
  "unittests."
  REPORTER
  XML
  OUTPUT_DIR
  .
  OUTPUT_PREFIX
  "unittests."
  OUTPUT_SUFFIX
  .xml)

# Add a file containing a set of constexpr tests
add_executable(constexpr_tests constexpr_tests.cpp)
target_link_libraries(constexpr_tests PRIVATE vkexec::vkexec_warnings vkexec::vkexec_options Catch2::Catch2WithMain)

vkexec_configure_linker(constexpr_tests)

catch_discover_tests(
  constexpr_tests
  TEST_PREFIX
  "constexpr."
  REPORTER
  XML
  OUTPUT_DIR
  .
  OUTPUT_PREFIX
  "constexpr."
  OUTPUT_SUFFIX
  .xml)

# Disable the constexpr portion of the test, and build again this allows us to have an executable that we can debug when
# things go wrong with the constexpr testing
add_executable(relaxed_constexpr_tests constexpr_tests.cpp)
target_link_libraries(relaxed_constexpr_tests PRIVATE vkexec::vkexec_warnings vkexec::vkexec_options
                                                      Catch2::Catch2WithMain)
target_compile_definitions(relaxed_constexpr_tests PRIVATE -DCATCH_CONFIG_RUNTIME_STATIC_REQUIRE)

vkexec_configure_linker(relaxed_constexpr_tests)

catch_discover_tests(
  relaxed_constexpr_tests
  TEST_PREFIX
  "relaxed_constexpr."
  REPORTER
  XML
  OUTPUT_DIR
  .
  OUTPUT_PREFIX
  "relaxed_constexpr."
  OUTPUT_SUFFIX
  .xml)
