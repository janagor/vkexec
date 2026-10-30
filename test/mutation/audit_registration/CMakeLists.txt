cmake_minimum_required(VERSION 3.28)
project(audit_registration LANGUAGES CXX)
include("${MUTATION_MODULE}")
if(vkexec_MULL_INTERNAL_SCOPE AND "$ENV{FAIL_STAGE}" STREQUAL "configure")
  message(FATAL_ERROR "Injected internal configure failure")
endif()
add_library(vkexec STATIC dummy.cpp)
add_executable(dummy dummy.cpp)
set(config "${CMAKE_CURRENT_SOURCE_DIR}/scope.yml")
set(source dummy.cpp)
set(name ranges)
set(test_target dummy)
if(CASE STREQUAL "missing_config")
  set(config missing.yml)
elseif(CASE STREQUAL "missing_target")
  set(test_target absent)
elseif(CASE STREQUAL "empty_sources")
  set(source)
elseif(CASE STREQUAL "invalid_name")
  set(name "bad-name")
endif()
vkexec_register_mutation_scope(
  NAME
  ${name}
  CONFIG
  "${config}"
  TEST_TARGET
  ${test_target}
  SOURCES
  ${source}
  TEST_ARGS
  "[ranges]"
  "--order"
  "lex"
  "--abort"
  "argument with spaces")
if(CASE STREQUAL "duplicate")
  vkexec_register_mutation_scope(
    NAME
    ranges
    CONFIG
    "${config}"
    TEST_TARGET
    dummy
    SOURCES
    dummy.cpp
    TEST_ARGS
    x)
endif()
if(CASE STREQUAL "relative_config")
  add_subdirectory(relative)
else()
  vkexec_register_mutation_scope(
    NAME
    other
    CONFIG
    "${CMAKE_CURRENT_SOURCE_DIR}/inactive.yml"
    TEST_TARGET
    dummy
    SOURCES
    dummy.cpp
    TEST_ARGS
    "other-argument")
endif()
if(CASE STREQUAL "integration")
  set(vkexec_MULL_PROFDATA "${CMAKE_CURRENT_SOURCE_DIR}/stub.py")
  set(vkexec_MULL_RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/stub.py")
  set(vkexec_MULL_TIMEOUT_MS 10000)
  set(vkexec_MULL_WORKERS 1)
  set(vkexec_MULL_MINIMUM_TIMEOUT_MS 250)
endif()
vkexec_finalize_mutation_scopes()
file(WRITE "${CMAKE_BINARY_DIR}/inherited-value.txt" "${vkexec_AUDIT_TEST_VALUE}")
foreach(target IN ITEMS vkexec dummy)
  get_target_property(options ${target} COMPILE_OPTIONS)
  file(WRITE "${CMAKE_BINARY_DIR}/${target}-options.txt" "${options}")
endforeach()
if(CASE STREQUAL "integration")
  # Exercise execution without loading a mutation plugin or profiling runtime.
  foreach(target IN ITEMS vkexec dummy)
    set_property(TARGET ${target} PROPERTY COMPILE_OPTIONS "")
    set_property(TARGET ${target} PROPERTY LINK_OPTIONS "")
    set_property(TARGET ${target} PROPERTY INTERFACE_LINK_OPTIONS "")
    set_property(TARGET ${target} PROPERTY CXX_COMPILER_LAUNCHER "")
  endforeach()
endif()

if(NOT vkexec_MULL_INTERNAL_SCOPE)
  file(APPEND "${CMAKE_BINARY_DIR}/mutation-configs/cache.cmake"
       "set(MUTATION_MODULE [==[${MUTATION_MODULE}]==] CACHE FILEPATH \"Fixture module\" FORCE)\n")
  # Preserve this fixture's mode in the generated private-build cache seed.
  file(APPEND "${CMAKE_BINARY_DIR}/mutation-configs/cache.cmake"
       "set(CASE [==[${CASE}]==] CACHE STRING \"Fixture mode\" FORCE)\n")
endif()
