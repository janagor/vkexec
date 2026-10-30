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
  # Presets may expand an unset MULL_RUNNER to an empty cache entry.
  # find_program treats that entry as a result and otherwise skips discovery.
  if("${vkexec_MULL_RUNNER}" STREQUAL "")
    unset(vkexec_MULL_RUNNER CACHE)
    unset(vkexec_MULL_RUNNER)
  endif()
  find_program(vkexec_MULL_RUNNER NAMES mull-runner-${clang_major} mull-runner REQUIRED)
  if(NOT EXISTS "${vkexec_MULL_RUNNER}" OR IS_DIRECTORY "${vkexec_MULL_RUNNER}")
    message(FATAL_ERROR "vkexec_MULL_RUNNER must point to an existing Mull runner executable")
  endif()
  find_program(vkexec_MULL_PROFDATA NAMES llvm-profdata-${clang_major} llvm-profdata REQUIRED)
  execute_process(COMMAND "${vkexec_MULL_PROFDATA}" --version OUTPUT_VARIABLE profdata_version COMMAND_ERROR_IS_FATAL
                                                                              ANY)
  if(NOT
     profdata_version
     MATCHES
     "LLVM version ${CMAKE_CXX_COMPILER_VERSION}([^0-9.]|$)")
    message(FATAL_ERROR "Mull requires llvm-profdata matching Clang ${CMAKE_CXX_COMPILER_VERSION}")
  endif()
  set(vkexec_MULL_TIMEOUT_MS
      "10000"
      CACHE STRING "Mull warmup/baseline timeout in milliseconds (mutant timeout is adaptive)")
  if(NOT
     vkexec_MULL_TIMEOUT_MS
     MATCHES
     "^[1-9][0-9]*$")
    message(FATAL_ERROR "vkexec_MULL_TIMEOUT_MS must be a positive integer")
  endif()
  set(vkexec_MULL_WORKERS
      "1"
      CACHE STRING "Number of concurrent Mull workers")
  set(vkexec_MULL_MINIMUM_TIMEOUT_MS
      "250"
      CACHE STRING "Minimum mutant timeout in milliseconds")
  foreach(setting IN ITEMS vkexec_MULL_WORKERS vkexec_MULL_MINIMUM_TIMEOUT_MS)
    if(NOT
       "${${setting}}"
       MATCHES
       "^[1-9][0-9]*$")
      message(FATAL_ERROR "${setting} must be a positive integer")
    endif()
  endforeach()
  cmake_path(
    CONVERT
    "$ENV{MULL_LIBRARY_PATH}"
    TO_CMAKE_PATH_LIST
    mull_library_paths)
  set(vkexec_MULL_LIBRARY_PATHS
      "${mull_library_paths}"
      CACHE STRING "Additional library directories for Mull's dependency scanner")
endfunction()

# Registrations are collected before creating all public audit targets.
function(vkexec_register_mutation_scope)
  cmake_parse_arguments(
    PARSE_ARGV
    0
    scope
    ""
    "NAME;CONFIG;TEST_TARGET"
    "SOURCES;TEST_ARGS")
  if(scope_UNPARSED_ARGUMENTS OR scope_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR "Invalid mutation scope arguments")
  endif()
  if(NOT
     scope_NAME
     MATCHES
     "^[a-z][a-z0-9_]*$")
    message(FATAL_ERROR "Invalid mutation scope name: ${scope_NAME}")
  endif()
  get_property(names GLOBAL PROPERTY VKEXEC_MUTATION_SCOPES)
  if(scope_NAME IN_LIST names)
    message(FATAL_ERROR "Duplicate mutation scope: ${scope_NAME}")
  endif()
  get_filename_component(
    scope_CONFIG
    "${scope_CONFIG}"
    REALPATH
    BASE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}")
  if(NOT EXISTS "${scope_CONFIG}" OR IS_DIRECTORY "${scope_CONFIG}")
    message(FATAL_ERROR "Missing mutation scope configuration: ${scope_CONFIG}")
  endif()
  if(NOT TARGET "${scope_TEST_TARGET}")
    message(FATAL_ERROR "Unknown mutation TEST_TARGET: ${scope_TEST_TARGET}")
  endif()
  get_target_property(target_type "${scope_TEST_TARGET}" TYPE)
  if(NOT
     target_type
     STREQUAL
     "EXECUTABLE")
    message(FATAL_ERROR "Mutation TEST_TARGET must be an executable")
  endif()
  if(NOT scope_SOURCES OR NOT scope_TEST_ARGS)
    message(FATAL_ERROR "Mutation scope requires SOURCES and TEST_ARGS")
  endif()
  set(normalized_sources)
  foreach(source IN LISTS scope_SOURCES)
    if(IS_ABSOLUTE "${source}"
       OR NOT EXISTS "${PROJECT_SOURCE_DIR}/${source}"
       OR IS_DIRECTORY "${PROJECT_SOURCE_DIR}/${source}")
      message(FATAL_ERROR "Invalid mutation source: ${source}")
    endif()
    file(REAL_PATH "${PROJECT_SOURCE_DIR}/${source}" resolved)
    file(
      RELATIVE_PATH
      relative
      "${PROJECT_SOURCE_DIR}"
      "${resolved}")
    if(relative MATCHES "^\\.\\.(/|$)" OR resolved IN_LIST normalized_sources)
      message(FATAL_ERROR "Outside-root or duplicate mutation source: ${source}")
    endif()
    list(APPEND normalized_sources "${resolved}")
  endforeach()
  set_property(GLOBAL APPEND PROPERTY VKEXEC_MUTATION_SCOPES "${scope_NAME}")
  foreach(
    field IN
    ITEMS CONFIG
          TEST_TARGET
          SOURCES
          TEST_ARGS)
    set_property(GLOBAL PROPERTY "VKEXEC_MUTATION_${scope_NAME}_${field}" "${scope_${field}}")
  endforeach()
endfunction()

function(_vkexec_instrument_mutation_target)
  cmake_parse_arguments(
    PARSE_ARGV
    0
    arg
    ""
    "TARGET;CONFIG"
    "")
  get_target_property(instrumented ${arg_TARGET} VKEXEC_MULL_INSTRUMENTED)
  if(instrumented)
    return()
  endif()
  set_property(TARGET ${arg_TARGET} PROPERTY VKEXEC_MULL_INSTRUMENTED TRUE)
  set(config "${arg_CONFIG}")
  file(SHA256 "${config}" config_hash)
  # A cached global launcher can survive vkexec_ENABLE_CACHE=OFF.
  # Always compile instrumented targets directly so plugin/config changes
  # cannot reuse objects produced by a compiler cache.
  set(mull_launcher
      "${CMAKE_COMMAND}"
      -E
      env
      "MULL_CONFIG=${config}")
  set_property(TARGET ${arg_TARGET} PROPERTY CXX_COMPILER_LAUNCHER "${mull_launcher}")
  # Regenerate mutants when configure_file detects a YAML change.
  target_compile_definitions(${arg_TARGET} PRIVATE "VKEXEC_MULL_CONFIG_HASH=\"${config_hash}\"")
  target_compile_options(
    ${arg_TARGET}
    PRIVATE -g
            -O0
            -grecord-command-line
            "-fpass-plugin=${vkexec_MULL_PLUGIN}"
            -fprofile-instr-generate
            -fcoverage-mapping)
  target_link_options(${arg_TARGET} PRIVATE -fprofile-instr-generate)
endfunction()

function(_vkexec_add_mutation_audit)
  cmake_parse_arguments(
    PARSE_ARGV
    0
    audit
    "INCLUDE_NOT_COVERED"
    "NAME;CONFIG;TEST_TARGET"
    "SOURCES;TEST_ARGS")
  set(config "${audit_CONFIG}")
  set(extra_args)
  if(audit_INCLUDE_NOT_COVERED)
    list(APPEND extra_args --include-not-covered)
  endif()
  set(library_args)
  foreach(directory IN LISTS vkexec_MULL_LIBRARY_PATHS)
    list(
      APPEND
      library_args
      --ld-search-path
      "${directory}")
  endforeach()
  set(expected_sources)
  foreach(source IN LISTS audit_SOURCES)
    list(APPEND expected_sources "${PROJECT_SOURCE_DIR}/${source}")
  endforeach()
  set(report_dir "${vkexec_MULL_REPORT_DIR}")
  set(report_name "vkexec_core_${audit_NAME}")
  set(profile_raw "${report_dir}/${report_name}.profraw")
  set(profile_data "${report_dir}/${report_name}.profdata")
  add_custom_target(
    vkexec_mutation_${audit_NAME}
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${report_dir}"
    COMMAND "${CMAKE_COMMAND}" -E rm -f "${report_dir}/${report_name}.sqlite" "${report_dir}/${report_name}.json"
            "${report_dir}/${report_name}.txt" "${profile_raw}" "${profile_data}"
    # Mull 0.35 misattributes multi-TU raw profiles. LLVM's indexed profile
    # restores coverage without changing the executable or bypassing filtering.
    # Catch2 3.12 defaults to random order. Stop on a detected assertion before
    # later tests do excessive work on mutated ranges; use the same order in
    # profile collection and evaluation. Timeouts remain distinct outcomes.
    COMMAND "${CMAKE_COMMAND}" -E env "LLVM_PROFILE_FILE=${profile_raw}" "$<TARGET_FILE:${audit_TEST_TARGET}>"
            ${audit_TEST_ARGS}
    COMMAND "${vkexec_MULL_PROFDATA}" merge -sparse "${profile_raw}" -o "${profile_data}"
    COMMAND
      "${CMAKE_COMMAND}" -E env "MULL_CONFIG=${config}" "${vkexec_MULL_RUNNER}" --allow-surviving --timeout
      "${vkexec_MULL_TIMEOUT_MS}" --workers "${vkexec_MULL_WORKERS}" --minimum-timeout
      "${vkexec_MULL_MINIMUM_TIMEOUT_MS}" ${library_args} ${extra_args} --coverage-info "${profile_data}" --reporters
      IDE --reporters SQLite --report-dir "${report_dir}" --report-name "${report_name}"
      "$<TARGET_FILE:${audit_TEST_TARGET}>" ${audit_TEST_ARGS}
    COMMAND
      "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/validate_mull_report.py"
      "${report_dir}/${report_name}.sqlite" ${expected_sources} --json-output "${report_dir}/${report_name}.json"
      --repository-root "${PROJECT_SOURCE_DIR}"
    DEPENDS ${audit_TEST_TARGET}
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
    USES_TERMINAL VERBATIM)
endfunction()

# Seed internal builds with the parent's toolchain/options, without sharing objects
# or dependency binary directories between scopes.
function(_vkexec_write_mutation_cache path)
  get_cmake_property(variables CACHE_VARIABLES)
  list(
    APPEND
    variables
    vkexec_MULL_PLUGIN
    vkexec_MULL_RUNNER
    vkexec_MULL_PROFDATA
    vkexec_MULL_TIMEOUT_MS
    vkexec_MULL_WORKERS
    vkexec_MULL_MINIMUM_TIMEOUT_MS)
  list(REMOVE_DUPLICATES variables)
  set(content "# Generated mutation build settings.\n")
  foreach(variable IN LISTS variables)
    if(variable MATCHES "^vkexec_"
       AND NOT
           variable
           MATCHES
           "(_SOURCE_DIR|_BINARY_DIR|MULL_INTERNAL_SCOPE|MULL_SCOPE|MULL_REPORT_DIR)$"
       OR variable
          MATCHES
          "^CMAKE_(C_COMPILER|CXX_COMPILER|AR|RANLIB|CXX_COMPILER_AR|CXX_COMPILER_RANLIB|MAKE_PROGRAM|BUILD_TYPE|TOOLCHAIN_FILE|SYSROOT|PREFIX_PATH|FIND_ROOT_PATH|C_FLAGS.*|CXX_FLAGS.*|EXE_LINKER_FLAGS.*|SHARED_LINKER_FLAGS.*|SKIP_INSTALL_RULES)$"
       OR variable MATCHES
          "^(BUILD_TESTING|BUILD_SHARED_LIBS|VKEXEC_STDEXEC_GIT_TAG|CPM_SOURCE_CACHE|FETCHCONTENT_SOURCE_DIR_.*)$")
      set(value "${${variable}}")
      set(delimiter "==")
      string(FIND "${value}" "]${delimiter}]" closing)
      while(
        NOT
        closing
        EQUAL
        -1)
        string(APPEND delimiter "=")
        string(FIND "${value}" "]${delimiter}]" closing)
      endwhile()
      string(
        APPEND content
        "set(${variable} [${delimiter}[${value}]${delimiter}] CACHE STRING \"Inherited mutation setting\" FORCE)\n")
    endif()
  endforeach()
  # Reuse downloaded dependency sources, never their build trees.
  foreach(package IN LISTS CPM_PACKAGES)
    string(TOUPPER "${package}" upper)
    set(source "${CPM_PACKAGE_${package}_SOURCE_DIR}")
    if(IS_DIRECTORY "${source}")
      string(APPEND content
             "set(FETCHCONTENT_SOURCE_DIR_${upper} [==[${source}]==] CACHE PATH \"Shared dependency source\" FORCE)\n")
    endif()
  endforeach()
  file(WRITE "${path}" "${content}")
endfunction()

function(vkexec_finalize_mutation_scopes)
  get_property(finalized GLOBAL PROPERTY VKEXEC_MUTATION_FINALIZED)
  if(finalized)
    message(FATAL_ERROR "Mutation scopes already finalized")
  endif()
  set_property(GLOBAL PROPERTY VKEXEC_MUTATION_FINALIZED TRUE)
  get_property(names GLOBAL PROPERTY VKEXEC_MUTATION_SCOPES)
  if(NOT names)
    message(FATAL_ERROR "No mutation scopes registered")
  endif()
  if(DEFINED vkexec_MULL_SCOPE)
    message(STATUS "vkexec_MULL_SCOPE is obsolete; all registered mutation targets are available")
    unset(vkexec_MULL_SCOPE CACHE)
  endif()
  if(vkexec_MULL_INTERNAL_SCOPE
     AND NOT
         vkexec_MULL_INTERNAL_SCOPE
         IN_LIST
         names)
    message(FATAL_ERROR "Unknown internal Mull scope '${vkexec_MULL_INTERNAL_SCOPE}'")
  endif()
  if(NOT vkexec_MULL_REPORT_DIR)
    set(vkexec_MULL_REPORT_DIR "${PROJECT_BINARY_DIR}/mutation-reports")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  foreach(script IN ITEMS report_validator coverage_diagnostics)
    if(script STREQUAL "report_validator")
      set(filename test_validate_mull_report.py)
    else()
      set(filename test_diagnose_mull_coverage.py)
    endif()
    add_test(NAME vkexec.mull_${script} COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/${filename}")
  endforeach()
  add_test(NAME vkexec.mull_audit_registration
           COMMAND "${Python3_EXECUTABLE}"
                   "${PROJECT_SOURCE_DIR}/test/mutation/audit_registration/test_registration.py")
  file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/mutation-configs")
  if(NOT vkexec_MULL_INTERNAL_SCOPE)
    _vkexec_write_mutation_cache("${PROJECT_BINARY_DIR}/mutation-configs/cache.cmake")
  endif()
  foreach(name IN LISTS names)
    foreach(
      field IN
      ITEMS CONFIG
            TEST_TARGET
            SOURCES
            TEST_ARGS)
      get_property(scope_${field} GLOBAL PROPERTY "VKEXEC_MUTATION_${name}_${field}")
    endforeach()
    set(config "${PROJECT_BINARY_DIR}/mutation-configs/${name}.yml")
    configure_file("${scope_CONFIG}" "${config}" COPYONLY)
    if(vkexec_MULL_INTERNAL_SCOPE)
      if(NOT
         name
         STREQUAL
         vkexec_MULL_INTERNAL_SCOPE)
        continue()
      endif()
      configure_file("${config}" "${PROJECT_BINARY_DIR}/mull.yml" COPYONLY)
      foreach(target IN ITEMS vkexec ${scope_TEST_TARGET})
        _vkexec_instrument_mutation_target(
          TARGET
          ${target}
          CONFIG
          "${PROJECT_BINARY_DIR}/mull.yml")
      endforeach()
      target_link_options(vkexec INTERFACE "$<BUILD_LOCAL_INTERFACE:-fprofile-instr-generate>")
    endif()
    foreach(variant IN ITEMS "" "_uncovered")
      if(vkexec_MULL_INTERNAL_SCOPE)
        set(extra)
        if(variant)
          set(extra INCLUDE_NOT_COVERED)
        endif()
        _vkexec_add_mutation_audit(
          NAME
          "${name}${variant}"
          CONFIG
          "${config}"
          TEST_TARGET
          "${scope_TEST_TARGET}"
          SOURCES
          ${scope_SOURCES}
          TEST_ARGS
          ${scope_TEST_ARGS}
          ${extra})
      else()
        add_custom_target(
          vkexec_mutation_${name}${variant}
          COMMAND
            "${CMAKE_COMMAND}" "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}" "-DBUILD_ROOT=${PROJECT_BINARY_DIR}"
            "-DSCOPE=${name}" "-DAUDIT=${name}${variant}" "-DGENERATOR=${CMAKE_GENERATOR}"
            "-DGENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}" "-DGENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            "-DCONFIG=$<CONFIG>" -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/RunMutationScope.cmake"
          USES_TERMINAL VERBATIM)
      endif()
    endforeach()
  endforeach()
endfunction()
