include_guard(GLOBAL)

function(vkexec_setup_mutation_testing)
  if(NOT PROJECT_IS_TOP_LEVEL OR NOT BUILD_TESTING)
    message(FATAL_ERROR "Mull requires a top-level build with BUILD_TESTING=ON")
  endif()
  if(NOT
     CMAKE_CXX_COMPILER_ID
     STREQUAL
     "Clang"
     OR NOT
        CMAKE_BUILD_TYPE
        STREQUAL
        "Debug")
    message(FATAL_ERROR "Mull requires Clang and CMAKE_BUILD_TYPE=Debug")
  endif()
  foreach(
    option IN
    ITEMS IPO
          COVERAGE
          HARDENING
          SANITIZER_ADDRESS
          SANITIZER_LEAK
          SANITIZER_UNDEFINED
          SANITIZER_THREAD
          SANITIZER_MEMORY)
    if(vkexec_ENABLE_${option})
      message(FATAL_ERROR "Disable vkexec_ENABLE_${option} in the dedicated Mull build")
    endif()
  endforeach()

  set(vkexec_MULL_PLUGIN
      ""
      CACHE FILEPATH "Mull IR frontend plugin built for the selected Clang")
  if(NOT EXISTS "${vkexec_MULL_PLUGIN}" OR IS_DIRECTORY "${vkexec_MULL_PLUGIN}")
    message(FATAL_ERROR "Set vkexec_MULL_PLUGIN to a Mull IR frontend matching your Clang version")
  endif()
  string(
    REGEX MATCH
          "^[0-9]+"
          clang_major
          "${CMAKE_CXX_COMPILER_VERSION}")
  find_program(vkexec_MULL_RUNNER NAMES mull-runner-${clang_major} REQUIRED)
  set(vkexec_MULL_TIMEOUT_MS
      "10000"
      CACHE STRING "Maximum duration of each Mull test run in milliseconds")
  if(NOT
     vkexec_MULL_TIMEOUT_MS
     MATCHES
     "^[1-9][0-9]*$")
    message(FATAL_ERROR "vkexec_MULL_TIMEOUT_MS must be a positive integer")
  endif()
  configure_file("${PROJECT_SOURCE_DIR}/mull.yml" "${PROJECT_BINARY_DIR}/mull.yml" COPYONLY)
endfunction()

function(vkexec_add_mutation_target)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME vkexec.mull_report_validator COMMAND "${Python3_EXECUTABLE}"
                                                     "${PROJECT_SOURCE_DIR}/scripts/test_validate_mull_report.py")
  set(config "${PROJECT_BINARY_DIR}/mull.yml")
  file(SHA256 "${config}" config_hash)
  foreach(target IN ITEMS vkexec vkexec_core_tests)
    # A cached global launcher can survive vkexec_ENABLE_CACHE=OFF.
    # Always compile instrumented targets directly so plugin/config changes
    # cannot reuse objects produced by a compiler cache.
    set(mull_launcher
        "${CMAKE_COMMAND}"
        -E
        env
        "MULL_CONFIG=${config}")
    set_property(TARGET ${target} PROPERTY CXX_COMPILER_LAUNCHER "${mull_launcher}")
    # Regenerate mutants when configure_file detects a YAML change.
    target_compile_definitions(${target} PRIVATE "VKEXEC_MULL_CONFIG_HASH=\"${config_hash}\"")
    target_compile_options(
      ${target}
      PRIVATE -g
              -O0
              -grecord-command-line
              "-fpass-plugin=${vkexec_MULL_PLUGIN}"
              -fprofile-instr-generate
              -fcoverage-mapping)
    target_link_options(${target} PRIVATE -fprofile-instr-generate)
  endforeach()
  # Other debug executables also link the instrumented core library.
  target_link_options(vkexec INTERFACE "$<BUILD_LOCAL_INTERFACE:-fprofile-instr-generate>")
  cmake_path(
    CONVERT
    "$ENV{MULL_LIBRARY_PATH}"
    TO_CMAKE_PATH_LIST
    mull_library_paths)
  set(vkexec_MULL_LIBRARY_PATHS
      "${mull_library_paths}"
      CACHE STRING "Additional library directories for Mull's dependency scanner")
  set(library_args)
  foreach(directory IN LISTS vkexec_MULL_LIBRARY_PATHS)
    list(
      APPEND
      library_args
      --ld-search-path
      "${directory}")
  endforeach()
  set(report_dir "${PROJECT_BINARY_DIR}/mutation-reports")
  foreach(audit IN ITEMS ranges ranges_uncovered)
    set(extra_args)
    if(audit STREQUAL "ranges_uncovered")
      list(APPEND extra_args --include-not-covered)
    endif()
    set(report_name "vkexec_core_${audit}")
    add_custom_target(
      vkexec_mutation_${audit}
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${report_dir}"
      COMMAND "${CMAKE_COMMAND}" -E rm -f "${report_dir}/${report_name}.sqlite" "${report_dir}/${report_name}.json"
              "${report_dir}/${report_name}.txt"
      COMMAND "$<TARGET_FILE:vkexec_core_tests>" "[ranges]"
      COMMAND
        "${CMAKE_COMMAND}" -E env "MULL_CONFIG=${config}" "${vkexec_MULL_RUNNER}" --allow-surviving --timeout
        "${vkexec_MULL_TIMEOUT_MS}" ${library_args} ${extra_args} --reporters IDE --reporters SQLite --report-dir
        "${report_dir}" --report-name "${report_name}" "$<TARGET_FILE:vkexec_core_tests>" "[ranges]"
      COMMAND
        "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/validate_mull_report.py"
        "${report_dir}/${report_name}.sqlite" "${PROJECT_SOURCE_DIR}/include/vkexec/detail/resource_ranges.hpp"
        --json-output "${report_dir}/${report_name}.json" --repository-root "${PROJECT_SOURCE_DIR}"
      DEPENDS vkexec_core_tests
      WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
      USES_TERMINAL VERBATIM)
  endforeach()
endfunction()
