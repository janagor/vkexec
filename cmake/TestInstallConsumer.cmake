cmake_minimum_required(VERSION 3.29)

if(NOT DEFINED VKEXEC_SOURCE_DIR OR
   NOT DEFINED VKEXEC_BINARY_DIR OR
   NOT DEFINED CONSUMER_SOURCE_DIR)
  message(FATAL_ERROR "Required test paths were not provided")
endif()

set(prefix "${VKEXEC_BINARY_DIR}/install-consumer/prefix")
set(build "${VKEXEC_BINARY_DIR}/install-consumer/build")

file(REMOVE_RECURSE "${prefix}" "${build}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${VKEXEC_BINARY_DIR}" --prefix "${prefix}"
  COMMAND_ERROR_IS_FATAL ANY)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${CONSUMER_SOURCE_DIR}" -B "${build}"
          "-DCMAKE_PREFIX_PATH=${prefix}" "-DCMAKE_BUILD_TYPE=Release"
  COMMAND_ERROR_IS_FATAL ANY)

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${build}" --config Release --parallel 10
  COMMAND_ERROR_IS_FATAL ANY)
