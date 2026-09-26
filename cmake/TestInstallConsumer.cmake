cmake_minimum_required(VERSION 3.29)

if(NOT DEFINED VKEXEC_SOURCE_DIR
   OR NOT DEFINED VKEXEC_BINARY_DIR
   OR NOT DEFINED CONSUMER_SOURCE_DIR
   OR NOT DEFINED VKEXEC_BOOST_SOURCE_DIR
   OR NOT DEFINED VKEXEC_GENERATOR)
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
set(boost_prefix "${VKEXEC_BINARY_DIR}/install-consumer/boost-prefix")
set(boost_build "${VKEXEC_BINARY_DIR}/install-consumer/boost-build")
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

file(
  REMOVE_RECURSE
  "${prefix}"
  "${build}"
  "${boost_prefix}"
  "${boost_build}")

set(generator_args "-G" "${VKEXEC_GENERATOR}")
if(DEFINED VKEXEC_GENERATOR_PLATFORM
   AND NOT
       "${VKEXEC_GENERATOR_PLATFORM}"
       STREQUAL
       "")
  list(
    APPEND
    generator_args
    "-A"
    "${VKEXEC_GENERATOR_PLATFORM}")
endif()
if(DEFINED VKEXEC_GENERATOR_TOOLSET
   AND NOT
       "${VKEXEC_GENERATOR_TOOLSET}"
       STREQUAL
       "")
  list(
    APPEND
    generator_args
    "-T"
    "${VKEXEC_GENERATOR_TOOLSET}")
endif()

vkexec_read_cache_entry("${parent_cache}" "CMAKE_CONFIGURATION_TYPES" _configuration_types)
vkexec_read_cache_entry("${parent_cache}" "CMAKE_TOOLCHAIN_FILE" _toolchain_file)
vkexec_read_cache_entry("${parent_cache}" "CMAKE_GENERATOR_INSTANCE" _generator_instance)
vkexec_read_cache_entry("${parent_cache}" "CMAKE_CXX_COMPILER" _cxx_compiler)

if(_toolchain_file AND NOT IS_ABSOLUTE "${_toolchain_file}")
  if(EXISTS "${VKEXEC_BINARY_DIR}/${_toolchain_file}")
    cmake_path(
      ABSOLUTE_PATH
      _toolchain_file
      BASE_DIRECTORY
      "${VKEXEC_BINARY_DIR}")
  elseif(EXISTS "${VKEXEC_SOURCE_DIR}/${_toolchain_file}")
    cmake_path(
      ABSOLUTE_PATH
      _toolchain_file
      BASE_DIRECTORY
      "${VKEXEC_SOURCE_DIR}")
  else()
    message(FATAL_ERROR "Could not resolve parent CMAKE_TOOLCHAIN_FILE: ${_toolchain_file}")
  endif()
endif()

set(boost_configure_args
    "-S"
    "${VKEXEC_BOOST_SOURCE_DIR}"
    "-B"
    "${boost_build}"
    ${generator_args}
    "-DBOOST_ENABLE_CMAKE=ON"
    "-DBOOST_INCLUDE_LIBRARIES=system"
    "-DBOOST_SKIP_INSTALL_RULES=OFF"
    "-DBUILD_TESTING=OFF"
    "-DCMAKE_INSTALL_PREFIX=${boost_prefix}")

set(consumer_configure_args
    "-S"
    "${CONSUMER_SOURCE_DIR}"
    "-B"
    "${build}"
    ${generator_args}
    "-DCMAKE_PREFIX_PATH=${prefix}"
    "-DBoost_ROOT=${boost_prefix}")

if(NOT _configuration_types)
  list(APPEND boost_configure_args "-DCMAKE_BUILD_TYPE=${VKEXEC_CONFIG}")
  list(APPEND consumer_configure_args "-DCMAKE_BUILD_TYPE=${VKEXEC_CONFIG}")
endif()

if(_toolchain_file)
  list(APPEND boost_configure_args "-DCMAKE_TOOLCHAIN_FILE=${_toolchain_file}")
  list(APPEND consumer_configure_args "-DCMAKE_TOOLCHAIN_FILE=${_toolchain_file}")
endif()

if(_generator_instance)
  list(APPEND boost_configure_args "-DCMAKE_GENERATOR_INSTANCE=${_generator_instance}")
  list(APPEND consumer_configure_args "-DCMAKE_GENERATOR_INSTANCE=${_generator_instance}")
endif()

if(NOT _toolchain_file
   AND _cxx_compiler
   AND NOT
       VKEXEC_GENERATOR
       MATCHES
       "Visual Studio"
   AND NOT
       VKEXEC_GENERATOR
       MATCHES
       "Xcode")
  list(APPEND boost_configure_args "-DCMAKE_CXX_COMPILER=${_cxx_compiler}")
  list(APPEND consumer_configure_args "-DCMAKE_CXX_COMPILER=${_cxx_compiler}")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" ${boost_configure_args} COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${boost_build}" --config "${VKEXEC_CONFIG}" --parallel 10
                        COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${boost_build}" --config "${VKEXEC_CONFIG}"
                        COMMAND_ERROR_IS_FATAL ANY)

execute_process(COMMAND "${CMAKE_COMMAND}" --install "${VKEXEC_BINARY_DIR}" --prefix "${prefix}" --config
                        "${VKEXEC_CONFIG}" COMMAND_ERROR_IS_FATAL ANY)

if(NOT EXISTS "${prefix}/include/vkexec/detail/stdexec_compat.hpp")
  message(FATAL_ERROR "Installed vkexec package is missing stdexec compatibility support header")
endif()

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
