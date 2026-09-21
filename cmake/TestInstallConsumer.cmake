cmake_minimum_required(VERSION 3.29)

if(NOT DEFINED VKEXEC_SOURCE_DIR OR
   NOT DEFINED VKEXEC_BINARY_DIR OR
   NOT DEFINED CONSUMER_SOURCE_DIR)
  message(FATAL_ERROR "Required test paths were not provided")
endif()

function(vkexec_read_cache_entry _cache_file _var _out)
  if(NOT EXISTS "${_cache_file}")
    return()
  endif()

  file(STRINGS "${_cache_file}" _cache_lines)
  foreach(_line IN LISTS _cache_lines)
    if(_line MATCHES "^${_var}:([^=]+)=(.+)$")
      set(${_out} "${CMAKE_MATCH_2}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

set(prefix "${VKEXEC_BINARY_DIR}/install-consumer/prefix")
set(build "${VKEXEC_BINARY_DIR}/install-consumer/build")
set(parent_cache "${VKEXEC_BINARY_DIR}/CMakeCache.txt")

file(REMOVE_RECURSE "${prefix}" "${build}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${VKEXEC_BINARY_DIR}" --prefix "${prefix}"
  COMMAND_ERROR_IS_FATAL ANY)

set(consumer_configure_args
    "-S" "${CONSUMER_SOURCE_DIR}"
    "-B" "${build}"
    "-DCMAKE_PREFIX_PATH=${prefix}"
    "-DCMAKE_BUILD_TYPE=Release")

vkexec_read_cache_entry("${parent_cache}" "Vulkan_INCLUDE_DIR" _vulkan_include_dir)
vkexec_read_cache_entry("${parent_cache}" "Vulkan_LIBRARY" _vulkan_library)
if(_vulkan_include_dir)
  list(APPEND consumer_configure_args "-DVulkan_INCLUDE_DIR=${_vulkan_include_dir}")
endif()
if(_vulkan_library)
  list(APPEND consumer_configure_args "-DVulkan_LIBRARY=${_vulkan_library}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" ${consumer_configure_args}
  COMMAND_ERROR_IS_FATAL ANY)

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${build}" --config Release --parallel 10
  COMMAND_ERROR_IS_FATAL ANY)
