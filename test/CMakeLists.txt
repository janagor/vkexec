# Workaround clang>=22 flagging __COUNTER__ as c2y-extension
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 22)
  add_definitions(-DCATCH_CONFIG_NO_COUNTER)
endif()

include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)

# CMake-script regression tests for cmake/Sanitizers.cmake (issue #141).
add_test(NAME cmake.sanitizers.apple_excludes_leak
         COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_sanitizers_apple_excludes_leak.cmake)
add_test(NAME cmake.sanitizers.non_apple_includes_leak
         COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_sanitizers_non_apple_includes_leak.cmake)
foreach(parent_standard 17 23)
  add_test(
    NAME vkexec.subdirectory_consumer.cxx${parent_standard}
    COMMAND
      "${CMAKE_COMMAND}" "-DVKEXEC_SOURCE_DIR=${vkexec_SOURCE_DIR}" "-DVKEXEC_BINARY_DIR=${vkexec_BINARY_DIR}"
      "-DVKEXEC_PARENT_STANDARD=${parent_standard}" "-DVKEXEC_CONFIG=$<CONFIG>" "-DVKEXEC_GENERATOR=${CMAKE_GENERATOR}"
      "-DVKEXEC_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}" "-DVKEXEC_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
      -P "${vkexec_SOURCE_DIR}/cmake/TestSubdirectoryConsumer.cmake")
endforeach()
set_tests_properties(vkexec.subdirectory_consumer.cxx17 vkexec.subdirectory_consumer.cxx23 PROPERTIES RUN_SERIAL TRUE)

option(vkexec_TEST_VALIDATION "Require Vulkan validation layers for unit tests" OFF)

function(vkexec_add_test_executable name module)
  cmake_parse_arguments(
    ARG
    ""
    ""
    "SOURCES;LIBRARIES"
    ${ARGN})
  add_executable(${name} ${ARG_SOURCES})
  set_target_properties(${name} PROPERTIES CXX_SCAN_FOR_MODULES OFF)
  target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/support")
  target_link_libraries(
    ${name}
    PRIVATE vkexec::vkexec_warnings
            vkexec::vkexec_options
            vkexec_exception_options
            ${ARG_LIBRARIES}
            Catch2::Catch2WithMain)
  if(vkexec_TEST_VALIDATION)
    target_compile_definitions(${name} PRIVATE VKEXEC_TEST_VALIDATION)
  endif()
  vkexec_configure_linker(${name})
  if(WIN32 AND BUILD_SHARED_LIBS)
    add_custom_command(
      TARGET ${name}
      PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:${name}> $<TARGET_FILE_DIR:${name}>
      COMMAND_EXPAND_LISTS)
  endif()
  catch_discover_tests(
    ${name}
    TEST_PREFIX
    "unittests.${module}."
    REPORTER
    XML
    OUTPUT_DIR
    .
    OUTPUT_PREFIX
    "unittests.${module}."
    OUTPUT_SUFFIX
    .xml)
  set_property(GLOBAL APPEND PROPERTY VKEXEC_TEST_EXECUTABLES ${name})
endfunction()

vkexec_add_test_executable(
  vkexec_core_tests
  core
  LIBRARIES
  vkexec::vkexec
  SOURCES
  core/tests.cpp
  core/barrier_header_tests.cpp
  core/sender_tests.cpp
  core/resource_table_tests.cpp
  core/descriptor_schema_tests.cpp
  core/pass_detail_tests.cpp
  core/dispatch_tests.cpp
  core/error_tests.cpp
  core/move_only_function_tests.cpp
  core/sampler_tests.cpp
  core/scheduler_tests.cpp
  core/resource_allocator_tests.cpp
  core/synchronization_tests.cpp)

vkexec_add_test_executable(
  vkexec_features_tests
  features
  LIBRARIES
  vkexec::features
  SOURCES
  features/feature_tests.cpp
  features/synchronization2_feature_tests.cpp
  features/context_capabilities_tests.cpp)

vkexec_add_test_executable(
  vkexec_graphics_tests
  graphics
  LIBRARIES
  vkexec::vkexec_graphics
  SOURCES
  graphics/graphics_tests.cpp)

vkexec_add_test_executable(
  vkexec_descriptor_heap_tests
  descriptor_heap
  LIBRARIES
  vkexec::ext_descriptor_heap
  SOURCES
  extensions/descriptor_heap/push_data_tests.cpp
  extensions/descriptor_heap/extension_tests.cpp
  extensions/descriptor_heap/resource_allocator_tests.cpp
  extensions/descriptor_heap/vulkan_requirements_tests.cpp)

vkexec_add_test_executable(
  vkexec_dynamic_rendering_tests
  dynamic_rendering
  LIBRARIES
  vkexec::ext_dynamic_rendering
  SOURCES
  extensions/dynamic_rendering/rendering_tests.cpp)

vkexec_add_test_executable(
  vkexec_timeline_tests
  timeline
  LIBRARIES
  vkexec::ext_timeline_semaphore
  SOURCES
  extensions/timeline_semaphore/timeline_semaphore_tests.cpp
  extensions/timeline_semaphore/frame_ring_tests.cpp)

vkexec_add_test_executable(
  vkexec_concurrency_tests
  concurrency
  LIBRARIES
  vkexec::vkexec
  SOURCES
  core/concurrency_tests.cpp)

if(vkexec_BUILD_VMA)
  vkexec_add_test_executable(
    vkexec_vma_tests
    vma
    LIBRARIES
    vkexec::vma
    SOURCES
    vma/buffer_tests.cpp
    vma/gpu_buffer_tests.cpp
    vma/image_tests.cpp
    vma/image_barrier_tests.cpp)
  vkexec_add_test_executable(
    vkexec_vma_features_tests
    vma_features
    LIBRARIES
    vkexec::vma
    vkexec::features
    SOURCES
    vma/features/tensor_tests.cpp
    vma/features/gpu_buffer_device_address_tests.cpp)
  vkexec_add_test_executable(
    vkexec_descriptor_heap_vma_tests
    descriptor_heap_vma
    LIBRARIES
    vkexec::ext_descriptor_heap
    vkexec::vma
    SOURCES
    integration/descriptor_heap_tests.cpp)
  vkexec_add_test_executable(
    vkexec_dynamic_rendering_vma_tests
    dynamic_rendering_vma
    LIBRARIES
    vkexec::ext_dynamic_rendering
    vkexec::vma
    SOURCES
    integration/rendering_tests.cpp)
endif()

if(vkexec_BUILD_TOOLS)
  vkexec_add_test_executable(
    vkexec_tools_tests
    tools
    LIBRARIES
    vkexec::tools
    SOURCES
    tools/spirv_compile_tests.cpp)
  if(vkexec_BUILD_VMA)
    vkexec_add_test_executable(
      vkexec_tools_vma_tests
      tools_vma
      LIBRARIES
      vkexec::tools
      vkexec::vma
      SOURCES
      integration/descriptor_heap_pipeline_tests.cpp
      integration/gpu_tests.cpp
      integration/tensor_pipeline_tests.cpp)
    vkexec_add_test_executable(
      vkexec_tools_graphics_vma_tests
      tools_graphics_vma
      LIBRARIES
      vkexec::tools
      vkexec::vma
      SOURCES
      integration/draw_stop_tests.cpp)
  endif()
endif()

# Compile each standalone public header in its own translation unit.
# Installed implementation-tail headers are excluded explicitly below.
function(vkexec_add_header_checks name library)
  set(sources)
  foreach(header IN LISTS ARGN)
    file(
      RELATIVE_PATH
      include_name
      "${vkexec_SOURCE_DIR}/include"
      "${header}")
    if(include_name MATCHES "^\\.\\.")
      message(FATAL_ERROR "Public header is outside vkexec include/: ${header}")
    endif()
    # This implementation tail is included after pass.hpp defines its step types.
    if(include_name STREQUAL "vkexec/detail/pass_lowering.hpp")
      continue()
    endif()
    string(MAKE_C_IDENTIFIER "${include_name}" source_name)
    set(source "${CMAKE_CURRENT_BINARY_DIR}/header_checks/${name}/${source_name}.cpp")
    file(
      GENERATE
      OUTPUT "${source}"
      CONTENT "#include <${include_name}>\n")
    list(APPEND sources "${source}")
  endforeach()
  add_library(${name} OBJECT ${sources})
  target_link_libraries(${name} PRIVATE ${library} vkexec_exception_options)
  set_target_properties(
    ${name}
    PROPERTIES CXX_SCAN_FOR_MODULES OFF
               UNITY_BUILD OFF
               DISABLE_PRECOMPILE_HEADERS ON
               CXX_CLANG_TIDY ""
               CXX_CPPCHECK "")
  set_property(GLOBAL APPEND PROPERTY VKEXEC_HEADER_CHECK_TARGETS ${name})
endfunction()

vkexec_add_header_checks(vkexec_core_header_checks vkexec::vkexec ${VKEXEC_CORE_HEADERS})
vkexec_add_header_checks(vkexec_features_header_checks vkexec::features ${VKEXEC_FEATURE_HEADERS})
vkexec_add_header_checks(vkexec_graphics_header_checks vkexec::vkexec_graphics ${VKEXEC_GRAPHICS_HEADERS})
vkexec_add_header_checks(vkexec_descriptor_heap_header_checks vkexec::ext_descriptor_heap
                         ${VKEXEC_DESCRIPTOR_HEAP_HEADERS})
vkexec_add_header_checks(vkexec_dynamic_rendering_header_checks vkexec::ext_dynamic_rendering
                         ${VKEXEC_DYNAMIC_RENDERING_HEADERS})
vkexec_add_header_checks(vkexec_timeline_header_checks vkexec::ext_timeline_semaphore ${VKEXEC_TIMELINE_HEADERS})
if(vkexec_BUILD_VMA)
  vkexec_add_header_checks(vkexec_vma_header_checks vkexec::vma ${VKEXEC_VMA_HEADERS})
endif()
if(vkexec_BUILD_TOOLS)
  vkexec_add_header_checks(vkexec_tools_header_checks vkexec::tools ${VKEXEC_TOOLS_HEADERS})
endif()

# Only this implementation test may include the private vk-bootstrap adapter.
add_executable(vk_bootstrap_feature_tests vk_bootstrap_feature_tests.cpp)
set_target_properties(vk_bootstrap_feature_tests PROPERTIES CXX_SCAN_FOR_MODULES OFF)
target_link_libraries(
  vk_bootstrap_feature_tests
  PRIVATE vkexec_exception_options
          vkexec::vkexec
          vkexec_private_headers
          vk-bootstrap::vk-bootstrap
          Catch2::Catch2WithMain)
catch_discover_tests(vk_bootstrap_feature_tests TEST_PREFIX "detail.")

get_property(vkexec_test_executables GLOBAL PROPERTY VKEXEC_TEST_EXECUTABLES)
get_property(vkexec_header_check_targets GLOBAL PROPERTY VKEXEC_HEADER_CHECK_TARGETS)
add_custom_target(vkexec_tests DEPENDS ${vkexec_test_executables} ${vkexec_header_check_targets}
                                       vk_bootstrap_feature_tests)
