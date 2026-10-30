cmake_minimum_required(VERSION 3.25)
project(mull_helper_attribution LANGUAGES CXX)
if(NOT
   CMAKE_CXX_COMPILER_ID
   STREQUAL
   "Clang"
   OR NOT
      CMAKE_CXX_COMPILER_VERSION
      VERSION_EQUAL
      "22.1.8")
  message(FATAL_ERROR "Use Clang 22.1.8 / Mull 0.35.0 from clang-mut")
endif()
set(MULL_PLUGIN
    "$ENV{MULL_PLUGIN}"
    CACHE FILEPATH "Mull frontend")
configure_file(mull.yml mull.yml COPYONLY)
add_executable(single single.cpp)
add_executable(single_if single.cpp)
target_compile_definitions(single_if PRIVATE ATTRIBUTION_IF)
add_executable(single_ordinary single.cpp)
add_executable(multi multi_main.cpp multi_a.cpp multi_b.cpp)
add_executable(multi_ordinary multi_main.cpp multi_a.cpp multi_b.cpp)
add_executable(out_of_line single.cpp out_of_line.cpp)
target_compile_definitions(single_ordinary PRIVATE ATTRIBUTION_ORDINARY)
target_compile_definitions(multi_ordinary PRIVATE ATTRIBUTION_ORDINARY)
target_compile_definitions(out_of_line PRIVATE ATTRIBUTION_OUT_OF_LINE)
foreach(
  target
  single
  single_if
  single_ordinary
  multi
  multi_ordinary
  out_of_line)
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
