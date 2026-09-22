cmake_minimum_required(VERSION 3.29)

if(NOT DEFINED VKEXEC_SOURCE_DIR
   OR NOT DEFINED VKEXEC_BINARY_DIR
   OR NOT DEFINED CONSUMER_SOURCE_DIR)
  message(FATAL_ERROR "Required test paths were not provided")
endif()

function(
  vkexec_read_cache_entry
  _cache_file
  _var
  _out)
  if(NOT EXISTS "${_cache_file}")
    return()
  endif()

  file(STRINGS "${_cache_file}" _cache_lines)
  foreach(_line IN LISTS _cache_lines)
    if(_line MATCHES "^${_var}:([^=]+)=(.+)$")
      set(${_out}
          "${CMAKE_MATCH_2}"
          PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

set(prefix "${VKEXEC_BINARY_DIR}/install-consumer/prefix")
set(build "${VKEXEC_BINARY_DIR}/install-consumer/build")
set(parent_cache "${VKEXEC_BINARY_DIR}/CMakeCache.txt")

# Multi-config generators place archives under $<CONFIG>/; cmake --install must
# select the same configuration that ctest/-C and the parent build used.
if(NOT DEFINED VKEXEC_CONFIG
   OR "${VKEXEC_CONFIG}" STREQUAL ""
   OR "${VKEXEC_CONFIG}" STREQUAL "$<CONFIG>")
  vkexec_read_cache_entry("${parent_cache}" "CMAKE_BUILD_TYPE" _build_type)
  vkexec_read_cache_entry("${parent_cache}" "CMAKE_DEFAULT_BUILD_TYPE" _default_build_type)
  if(_build_type)
    set(VKEXEC_CONFIG "${_build_type}")
  elseif(_default_build_type)
    set(VKEXEC_CONFIG "${_default_build_type}")
  else()
    set(VKEXEC_CONFIG "Release")
  endif()
endif()

file(REMOVE_RECURSE "${prefix}" "${build}")

execute_process(COMMAND "${CMAKE_COMMAND}" --install "${VKEXEC_BINARY_DIR}" --prefix "${prefix}" --config
                        "${VKEXEC_CONFIG}" COMMAND_ERROR_IS_FATAL ANY)

set(consumer_configure_args
    "-S"
    "${CONSUMER_SOURCE_DIR}"
    "-B"
    "${build}"
    "-DCMAKE_PREFIX_PATH=${prefix}"
    "-DCMAKE_BUILD_TYPE=${VKEXEC_CONFIG}")

vkexec_read_cache_entry("${parent_cache}" "Vulkan_INCLUDE_DIR" _vulkan_include_dir)
vkexec_read_cache_entry("${parent_cache}" "Vulkan_LIBRARY" _vulkan_library)
if(_vulkan_include_dir)
  list(APPEND consumer_configure_args "-DVulkan_INCLUDE_DIR=${_vulkan_include_dir}")
endif()
if(_vulkan_library)
  list(APPEND consumer_configure_args "-DVulkan_LIBRARY=${_vulkan_library}")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" ${consumer_configure_args} COMMAND_ERROR_IS_FATAL ANY)

execute_process(COMMAND "${CMAKE_COMMAND}" --build "${build}" --config "${VKEXEC_CONFIG}" --parallel 10
                        COMMAND_ERROR_IS_FATAL ANY)
