cmake_minimum_required(VERSION 3.25)
project(mull_coverage_attribution LANGUAGES CXX)
if(NOT
   CMAKE_CXX_COMPILER_ID
   STREQUAL
   "Clang"
   OR NOT
      CMAKE_CXX_COMPILER_VERSION
      VERSION_EQUAL
      "22.1.8")
  message(FATAL_ERROR "Use the clang-mut shell (Clang 22.1.8 / Mull 0.35.0)")
endif()
set(MULL_PLUGIN
    "$ENV{MULL_PLUGIN}"
    CACHE FILEPATH "Mull 22 frontend")
configure_file(mull.yml mull.yml COPYONLY)
add_executable(single main.cpp)
add_executable(multi main.cpp a.cpp b.cpp)
target_compile_definitions(multi PRIVATE ATTRIBUTION_MULTI_TU)
add_executable(
  out_of_line
  main.cpp
  a.cpp
  b.cpp
  out_of_line.cpp)
target_compile_definitions(out_of_line PRIVATE ATTRIBUTION_MULTI_TU ATTRIBUTION_OUT_OF_LINE)
foreach(target single multi out_of_line)
  set_property(TARGET ${target} PROPERTY CXX_COMPILER_LAUNCHER
                                         "${CMAKE_COMMAND};-E;env;MULL_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/mull.yml")
  target_compile_options(
    ${target}
    PRIVATE -g
            -O0
            -grecord-command-line
            "-fpass-plugin=${MULL_PLUGIN}"
            -fprofile-instr-generate
            -fcoverage-mapping)
  target_link_options(${target} PRIVATE -fprofile-instr-generate)
endforeach()
