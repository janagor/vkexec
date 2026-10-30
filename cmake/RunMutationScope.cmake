# Mull 0.35 cannot isolate uncovered mutants with runtime path filters. Keep one
# public build entry point and persistent private instrumented builds per scope.
file(MAKE_DIRECTORY "${BUILD_ROOT}/mutation-builds")
# Also serialize targets launched from different build processes. Timeouts should
# measure one audit at a time, and covered/uncovered targets share each scope's tree.
file(LOCK "${BUILD_ROOT}/mutation-builds/audit.lock" GUARD PROCESS)
set(build "${BUILD_ROOT}/mutation-builds/${SCOPE}")
set(generator_args -G "${GENERATOR}")
if(GENERATOR_PLATFORM)
  list(
    APPEND
    generator_args
    -A
    "${GENERATOR_PLATFORM}")
endif()
if(GENERATOR_TOOLSET)
  list(
    APPEND
    generator_args
    -T
    "${GENERATOR_TOOLSET}")
endif()
execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${build}" ${generator_args} -C
    "${BUILD_ROOT}/mutation-configs/cache.cmake" "-Dvkexec_MULL_INTERNAL_SCOPE:INTERNAL=${SCOPE}"
    "-Dvkexec_MULL_REPORT_DIR:INTERNAL=${BUILD_ROOT}/mutation-reports" COMMAND_ERROR_IS_FATAL ANY)
# Respect an explicit build-parallelism setting; bound memory use by default.
set(jobs "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}")
if(NOT jobs)
  set(jobs 2)
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${build}" --config "${CONFIG}" --parallel "${jobs}" --target
                        "vkexec_mutation_${AUDIT}" COMMAND_ERROR_IS_FATAL ANY)
