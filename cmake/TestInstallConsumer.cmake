cmake_minimum_required(VERSION 3.29)

if(NOT DEFINED VKEXEC_SOURCE_DIR
   OR NOT DEFINED VKEXEC_BINARY_DIR
   OR NOT DEFINED CONSUMER_SOURCE_DIR
   OR NOT DEFINED VKEXEC_GENERATOR)
  message(FATAL_ERROR "Required test paths were not provided")
endif()

string(SHA256 package_contract_id "${VKEXEC_BINARY_DIR}")
string(
  SUBSTRING "${package_contract_id}"
            0
            12
            package_contract_id)
cmake_path(
  GET
  VKEXEC_SOURCE_DIR
  PARENT_PATH
  package_contract_parent)
set(test_root "${package_contract_parent}/package-contract-${package_contract_id}")
set(stage_prefix "${test_root}/stage")
set(relocated_prefix "${test_root}/relocated")
set(consumer_source "${test_root}/consumer-src")
set(consumer_build "${test_root}/consumer-build")
file(REMOVE_RECURSE "${test_root}")
file(MAKE_DIRECTORY "${test_root}")
file(COPY "${CONSUMER_SOURCE_DIR}/" DESTINATION "${consumer_source}")

if(NOT DEFINED VKEXEC_CONFIG
   OR "${VKEXEC_CONFIG}" STREQUAL ""
   OR "${VKEXEC_CONFIG}" STREQUAL "$<CONFIG>")
  set(VKEXEC_CONFIG Release)
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${VKEXEC_BINARY_DIR}" --config "${VKEXEC_CONFIG}" --parallel
                        COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${VKEXEC_BINARY_DIR}" --prefix "${stage_prefix}" --config
                        "${VKEXEC_CONFIG}" COMMAND_ERROR_IS_FATAL ANY)
file(RENAME "${stage_prefix}" "${relocated_prefix}")

file(GLOB_RECURSE package_metadata "${relocated_prefix}/*/cmake/vkexec/*.cmake")
if(NOT package_metadata)
  message(FATAL_ERROR "vkexec package metadata was not installed")
endif()
foreach(metadata_file IN LISTS package_metadata)
  file(READ "${metadata_file}" metadata)
  foreach(
    forbidden IN
    ITEMS "${VKEXEC_SOURCE_DIR}"
          "${VKEXEC_BINARY_DIR}"
          "/_deps/"
          "CPM")
    string(FIND "${metadata}" "${forbidden}" leak_position)
    if(NOT
       leak_position
       EQUAL
       -1)
      message(FATAL_ERROR "${metadata_file} contains a source, build, or CPM path: ${forbidden}")
    endif()
  endforeach()
endforeach()

set(configure_args
    -S
    "${consumer_source}"
    -B
    "${consumer_build}"
    -G
    "${VKEXEC_GENERATOR}"
    "-DCMAKE_PREFIX_PATH=${relocated_prefix}"
    "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF"
    "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF"
    "-DVKEXEC_PREFIX=${relocated_prefix}"
    "-DVKEXEC_FORBIDDEN_SOURCE=${VKEXEC_SOURCE_DIR}"
    "-DVKEXEC_FORBIDDEN_BUILD=${VKEXEC_BINARY_DIR}"
    "-DVKEXEC_EXPECT_VMA=${VKEXEC_EXPECT_VMA}"
    "-DVKEXEC_EXPECT_TOOLS=${VKEXEC_EXPECT_TOOLS}")
if(VKEXEC_GENERATOR_PLATFORM)
  list(
    APPEND
    configure_args
    -A
    "${VKEXEC_GENERATOR_PLATFORM}")
endif()
if(VKEXEC_GENERATOR_TOOLSET)
  list(
    APPEND
    configure_args
    -T
    "${VKEXEC_GENERATOR_TOOLSET}")
endif()
if(VKEXEC_GENERATOR_INSTANCE)
  list(APPEND configure_args "-DCMAKE_GENERATOR_INSTANCE=${VKEXEC_GENERATOR_INSTANCE}")
endif()
if(NOT VKEXEC_GENERATOR_IS_MULTI_CONFIG)
  list(APPEND configure_args "-DCMAKE_BUILD_TYPE=${VKEXEC_CONFIG}")
endif()
set(vkexec_consumer_toolchain "${VKEXEC_TOOLCHAIN_FILE}")
if(vkexec_consumer_toolchain AND NOT IS_ABSOLUTE "${vkexec_consumer_toolchain}")
  if(EXISTS "${VKEXEC_BINARY_DIR}/${vkexec_consumer_toolchain}")
    cmake_path(
      ABSOLUTE_PATH
      vkexec_consumer_toolchain
      BASE_DIRECTORY
      "${VKEXEC_BINARY_DIR}")
  elseif(EXISTS "${VKEXEC_SOURCE_DIR}/${vkexec_consumer_toolchain}")
    cmake_path(
      ABSOLUTE_PATH
      vkexec_consumer_toolchain
      BASE_DIRECTORY
      "${VKEXEC_SOURCE_DIR}")
  else()
    message(FATAL_ERROR "Could not resolve toolchain file: ${vkexec_consumer_toolchain}")
  endif()
endif()
if(vkexec_consumer_toolchain)
  list(APPEND configure_args "-DCMAKE_TOOLCHAIN_FILE=${vkexec_consumer_toolchain}")
endif()
if(NOT vkexec_consumer_toolchain
   AND VKEXEC_CXX_COMPILER
   AND NOT
       VKEXEC_GENERATOR
       MATCHES
       "Visual Studio|Xcode")
  list(APPEND configure_args "-DCMAKE_CXX_COMPILER=${VKEXEC_CXX_COMPILER}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" ${configure_args} COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${consumer_build}" --config "${VKEXEC_CONFIG}" --parallel 10
                        COMMAND_ERROR_IS_FATAL ANY)
