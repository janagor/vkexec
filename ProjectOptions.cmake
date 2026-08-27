include(cmake/LibFuzzer.cmake)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)

include(CheckCXXSourceCompiles)

macro(vkexec_supports_sanitizers)
  # Emscripten doesn't support sanitizers
  if(EMSCRIPTEN)
    set(SUPPORTS_UBSAN OFF)
    set(SUPPORTS_ASAN OFF)
  elseif((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND NOT WIN32)

    message(STATUS "Sanity checking UndefinedBehaviorSanitizer, it should be supported on this platform")
    set(TEST_PROGRAM "int main() { return 0; }")

    # Check if UndefinedBehaviorSanitizer works at link time
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=undefined")
    set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=undefined")
    check_cxx_source_compiles("${TEST_PROGRAM}" HAS_UBSAN_LINK_SUPPORT)

    if(HAS_UBSAN_LINK_SUPPORT)
      message(STATUS "UndefinedBehaviorSanitizer is supported at both compile and link time.")
      set(SUPPORTS_UBSAN ON)
    else()
      message(WARNING "UndefinedBehaviorSanitizer is NOT supported at link time.")
      set(SUPPORTS_UBSAN OFF)
    endif()
  else()
    set(SUPPORTS_UBSAN OFF)
  endif()

  if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND WIN32)
    set(SUPPORTS_ASAN OFF)
  else()
    if(NOT WIN32)
      message(STATUS "Sanity checking AddressSanitizer, it should be supported on this platform")
      set(TEST_PROGRAM "int main() { return 0; }")

      # Check if AddressSanitizer works at link time
      set(CMAKE_REQUIRED_FLAGS "-fsanitize=address")
      set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=address")
      check_cxx_source_compiles("${TEST_PROGRAM}" HAS_ASAN_LINK_SUPPORT)

      if(HAS_ASAN_LINK_SUPPORT)
        message(STATUS "AddressSanitizer is supported at both compile and link time.")
        set(SUPPORTS_ASAN ON)
      else()
        message(WARNING "AddressSanitizer is NOT supported at link time.")
        set(SUPPORTS_ASAN OFF)
      endif()
    else()
      set(SUPPORTS_ASAN ON)
    endif()
  endif()
endmacro()

macro(vkexec_setup_options)
  option(VKEXEC_ENABLE_EXCEPTIONS "Enable C++ exceptions for vkexec targets" OFF)
  option(vkexec_ENABLE_HARDENING "Enable hardening" ON)
  option(vkexec_ENABLE_COVERAGE "Enable coverage reporting" OFF)
  cmake_dependent_option(
    vkexec_ENABLE_GLOBAL_HARDENING
    "Attempt to push hardening options to built dependencies"
    ON
    vkexec_ENABLE_HARDENING
    OFF)

  vkexec_supports_sanitizers()

  if(NOT PROJECT_IS_TOP_LEVEL OR vkexec_PACKAGING_MAINTAINER_MODE)
    option(vkexec_ENABLE_IPO "Enable IPO/LTO" OFF)
    option(vkexec_WARNINGS_AS_ERRORS "Treat Warnings As Errors" OFF)
    option(vkexec_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
    option(vkexec_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
    option(vkexec_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" OFF)
    option(vkexec_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
    option(vkexec_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
    option(vkexec_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
    option(vkexec_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
    option(vkexec_ENABLE_CPPCHECK "Enable cpp-check analysis" OFF)
    option(vkexec_ENABLE_PCH "Enable precompiled headers" OFF)
    option(vkexec_ENABLE_CACHE "Enable ccache" OFF)
    option(vkexec_BUILD_EXAMPLES "Build example executables" OFF)
  else()
    option(vkexec_ENABLE_IPO "Enable IPO/LTO" ON)
    option(vkexec_WARNINGS_AS_ERRORS "Treat Warnings As Errors" ON)
    option(vkexec_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" ${SUPPORTS_ASAN})
    option(vkexec_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
    option(vkexec_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" ${SUPPORTS_UBSAN})
    option(vkexec_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
    option(vkexec_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
    option(vkexec_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
    option(vkexec_ENABLE_CLANG_TIDY "Enable clang-tidy" ON)
    option(vkexec_ENABLE_CPPCHECK "Enable cpp-check analysis" ON)
    option(vkexec_ENABLE_PCH "Enable precompiled headers" OFF)
    option(vkexec_ENABLE_CACHE "Enable ccache" ON)
    option(vkexec_BUILD_EXAMPLES "Build example executables" ON)
  endif()

  if(NOT PROJECT_IS_TOP_LEVEL)
    mark_as_advanced(
      vkexec_ENABLE_IPO
      vkexec_WARNINGS_AS_ERRORS
      vkexec_ENABLE_SANITIZER_ADDRESS
      vkexec_ENABLE_SANITIZER_LEAK
      vkexec_ENABLE_SANITIZER_UNDEFINED
      vkexec_ENABLE_SANITIZER_THREAD
      vkexec_ENABLE_SANITIZER_MEMORY
      vkexec_ENABLE_UNITY_BUILD
      vkexec_ENABLE_CLANG_TIDY
      vkexec_ENABLE_CPPCHECK
      vkexec_ENABLE_COVERAGE
      vkexec_ENABLE_PCH
      vkexec_ENABLE_CACHE)
  endif()

  vkexec_check_libfuzzer_support(LIBFUZZER_SUPPORTED)
  if(LIBFUZZER_SUPPORTED
     AND (vkexec_ENABLE_SANITIZER_ADDRESS
          OR vkexec_ENABLE_SANITIZER_THREAD
          OR vkexec_ENABLE_SANITIZER_UNDEFINED))
    set(DEFAULT_FUZZER ON)
  else()
    set(DEFAULT_FUZZER OFF)
  endif()

  option(vkexec_BUILD_FUZZ_TESTS "Enable fuzz testing executable" ${DEFAULT_FUZZER})

endmacro()

macro(vkexec_global_options)
  if(vkexec_ENABLE_IPO)
    include(cmake/InterproceduralOptimization.cmake)
    vkexec_enable_ipo()
  endif()

  vkexec_supports_sanitizers()

  if(vkexec_ENABLE_HARDENING AND vkexec_ENABLE_GLOBAL_HARDENING)
    include(cmake/Hardening.cmake)
    if(NOT SUPPORTS_UBSAN
       OR vkexec_ENABLE_SANITIZER_UNDEFINED
       OR vkexec_ENABLE_SANITIZER_ADDRESS
       OR vkexec_ENABLE_SANITIZER_THREAD
       OR vkexec_ENABLE_SANITIZER_LEAK)
      set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
    else()
      set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
    endif()
    message("${vkexec_ENABLE_HARDENING} ${ENABLE_UBSAN_MINIMAL_RUNTIME} ${vkexec_ENABLE_SANITIZER_UNDEFINED}")
    vkexec_enable_hardening(vkexec_options ON ${ENABLE_UBSAN_MINIMAL_RUNTIME})
  endif()
endmacro()

macro(vkexec_local_options)
  if(PROJECT_IS_TOP_LEVEL)
    include(cmake/StandardProjectSettings.cmake)
  endif()

  add_library(vkexec_warnings INTERFACE)
  add_library(vkexec_options INTERFACE)

  include(cmake/CompilerWarnings.cmake)
  vkexec_set_project_warnings(
    vkexec_warnings
    ${vkexec_WARNINGS_AS_ERRORS}
    ""
    ""
    ""
    "")

  include(cmake/Linker.cmake)
  # Must configure each target with linker options, we're avoiding setting it globally for now

  if(NOT EMSCRIPTEN)
    include(cmake/Sanitizers.cmake)
    vkexec_enable_sanitizers(
      vkexec_options
      ${vkexec_ENABLE_SANITIZER_ADDRESS}
      ${vkexec_ENABLE_SANITIZER_LEAK}
      ${vkexec_ENABLE_SANITIZER_UNDEFINED}
      ${vkexec_ENABLE_SANITIZER_THREAD}
      ${vkexec_ENABLE_SANITIZER_MEMORY})
  endif()

  set_target_properties(vkexec_options PROPERTIES UNITY_BUILD ${vkexec_ENABLE_UNITY_BUILD})

  if(vkexec_ENABLE_PCH)
    target_precompile_headers(
      vkexec_options
      INTERFACE
      <vector>
      <string>
      <utility>)
  endif()

  if(vkexec_ENABLE_CACHE)
    include(cmake/Cache.cmake)
    vkexec_enable_cache()
  endif()

  include(cmake/StaticAnalyzers.cmake)
  if(vkexec_ENABLE_CLANG_TIDY)
    vkexec_enable_clang_tidy(vkexec_options ${vkexec_WARNINGS_AS_ERRORS})
  endif()

  if(vkexec_ENABLE_CPPCHECK)
    vkexec_enable_cppcheck(${vkexec_WARNINGS_AS_ERRORS} "" # override cppcheck options
    )
  endif()

  if(vkexec_ENABLE_COVERAGE)
    include(cmake/Tests.cmake)
    vkexec_enable_coverage(vkexec_options)
  endif()

  if(vkexec_WARNINGS_AS_ERRORS)
    check_cxx_compiler_flag("-Wl,--fatal-warnings" LINKER_FATAL_WARNINGS)
    if(LINKER_FATAL_WARNINGS)
      # This is not working consistently, so disabling for now
      # target_link_options(vkexec_options INTERFACE -Wl,--fatal-warnings)
    endif()
  endif()

  if(vkexec_ENABLE_HARDENING AND NOT vkexec_ENABLE_GLOBAL_HARDENING)
    include(cmake/Hardening.cmake)
    if(NOT SUPPORTS_UBSAN
       OR vkexec_ENABLE_SANITIZER_UNDEFINED
       OR vkexec_ENABLE_SANITIZER_ADDRESS
       OR vkexec_ENABLE_SANITIZER_THREAD
       OR vkexec_ENABLE_SANITIZER_LEAK)
      set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
    else()
      set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
    endif()
    vkexec_enable_hardening(vkexec_options OFF ${ENABLE_UBSAN_MINIMAL_RUNTIME})
  endif()

  if(NOT VKEXEC_ENABLE_EXCEPTIONS)
    if(MSVC)
      target_compile_options(vkexec_options INTERFACE /EHsc- /D_HAS_EXCEPTIONS=0)
    else()
      target_compile_options(vkexec_options INTERFACE -fno-exceptions)
    endif()
  endif()

endmacro()
