cmake_minimum_required(VERSION 3.29)

foreach(
  required
  VKEXEC_SOURCE_DIR
  VKEXEC_BINARY_DIR
  VKEXEC_GENERATOR
  VKEXEC_PARENT_STANDARD)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()

set(parent_cache "${VKEXEC_BINARY_DIR}/CMakeCache.txt")
set(build "${VKEXEC_BINARY_DIR}/subdirectory-consumer-${VKEXEC_PARENT_STANDARD}")

function(vkexec_read_cache_entry variable output)
  file(STRINGS "${parent_cache}" lines REGEX "^${variable}:[^=]+=")
  if(lines)
    list(
      GET
      lines
      0
      line)
    string(
      REGEX
      REPLACE "^[^=]+="
              ""
              value
              "${line}")
    set(${output}
        "${value}"
        PARENT_SCOPE)
  endif()
endfunction()

set(configure_args
    -S
    "${VKEXEC_SOURCE_DIR}/test/fixtures/subdirectory_consumer"
    -B
    "${build}"
    -G
    "${VKEXEC_GENERATOR}"
    "-DCMAKE_CXX_STANDARD=${VKEXEC_PARENT_STANDARD}"
    -DBUILD_TESTING=OFF
    -Dvkexec_BUILD_VMA=OFF
    -Dvkexec_BUILD_TOOLS=OFF
    -Dvkexec_BUILD_EXAMPLES=OFF)

if(DEFINED VKEXEC_GENERATOR_PLATFORM
   AND NOT
       "${VKEXEC_GENERATOR_PLATFORM}"
       STREQUAL
       "")
  list(
    APPEND
    configure_args
    -A
    "${VKEXEC_GENERATOR_PLATFORM}")
endif()
if(DEFINED VKEXEC_GENERATOR_TOOLSET
   AND NOT
       "${VKEXEC_GENERATOR_TOOLSET}"
       STREQUAL
       "")
  list(
    APPEND
    configure_args
    -T
    "${VKEXEC_GENERATOR_TOOLSET}")
endif()

foreach(
  variable
  CMAKE_TOOLCHAIN_FILE
  CMAKE_GENERATOR_INSTANCE
  CMAKE_MAKE_PROGRAM
  CMAKE_CXX_COMPILER
  CMAKE_BUILD_TYPE
  Vulkan_INCLUDE_DIR
  Vulkan_LIBRARY)
  if(variable STREQUAL "CMAKE_CXX_COMPILER" AND VKEXEC_GENERATOR MATCHES "Visual Studio|Xcode")
    continue()
  endif()
  vkexec_read_cache_entry("${variable}" value)
  if(value)
    if(variable STREQUAL "CMAKE_TOOLCHAIN_FILE" AND NOT IS_ABSOLUTE "${value}")
      cmake_path(
        ABSOLUTE_PATH
        value
        BASE_DIRECTORY
        "${VKEXEC_BINARY_DIR}")
    endif()
    list(APPEND configure_args "-D${variable}=${value}")
  endif()
  unset(value)
endforeach()

vkexec_read_cache_entry(CPM_SOURCE_CACHE cpm_source_cache)
if(NOT cpm_source_cache)
  if(DEFINED ENV{CPM_SOURCE_CACHE}
     AND NOT
         "$ENV{CPM_SOURCE_CACHE}"
         STREQUAL
         "")
    set(cpm_source_cache "$ENV{CPM_SOURCE_CACHE}")
  else()
    set(cpm_source_cache "${VKEXEC_BINARY_DIR}/cpm-source-cache")
  endif()
endif()
if(NOT IS_ABSOLUTE "${cpm_source_cache}")
  cmake_path(
    ABSOLUTE_PATH
    cpm_source_cache
    BASE_DIRECTORY
    "${VKEXEC_BINARY_DIR}")
endif()
list(APPEND configure_args "-DCPM_SOURCE_CACHE=${cpm_source_cache}")

execute_process(COMMAND "${CMAKE_COMMAND}" ${configure_args} COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${build}" --target vkexec_subdirectory_consumer vkexec_unrelated
                        --config "${VKEXEC_CONFIG}" --parallel 10 COMMAND_ERROR_IS_FATAL ANY)
