option(vkexec_RECOMPILE_SAMPLE_SHADERS "Recompile Vulkan sample shaders" OFF)

if(vkexec_RECOMPILE_SAMPLE_SHADERS)
  find_package(Vulkan QUIET)
  if(TARGET Vulkan::glslc)
    set(VKEXEC_GLSLC Vulkan::glslc)
  elseif(Vulkan_GLSLC_EXECUTABLE)
    set(VKEXEC_GLSLC "${Vulkan_GLSLC_EXECUTABLE}")
  else()
    find_program(
      VKEXEC_GLSLC
      NAMES glslc
      HINTS "$ENV{VULKAN_SDK}/bin" REQUIRED)
  endif()
endif()

function(vkexec_add_sample_shaders target)
  set(shader_outputs)
  foreach(shader IN LISTS ARGN)
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}")
    set(committed "${source}.spv")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/shaders/${shader}.spv")
    if(vkexec_RECOMPILE_SAMPLE_SHADERS)
      set(depfile "${output}.d")
      add_custom_command(
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMAND "${VKEXEC_GLSLC}" --target-env=vulkan1.3 -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders" -MD -MF "${depfile}"
                "${source}" -o "${output}"
        DEPFILE "${depfile}"
        DEPENDS "${source}"
        VERBATIM)
    else()
      if(NOT EXISTS "${committed}")
        message(FATAL_ERROR "Missing committed sample shader: ${committed}")
      endif()
      add_custom_command(
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${committed}" "${output}"
        DEPENDS "${committed}"
        VERBATIM)
    endif()
    list(APPEND shader_outputs "${output}")
  endforeach()
  add_custom_target(${target}_shaders DEPENDS ${shader_outputs})
  add_dependencies(${target} ${target}_shaders)
  target_compile_definitions(${target} PRIVATE "VKEXEC_SAMPLE_SHADER_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/shaders\"")
endfunction()

add_subdirectory(hello_triangle_1_3)
add_subdirectory(compute_nbody)
add_subdirectory(pipeline_barriers)
add_subdirectory(async_compute)
add_subdirectory(timeline_semaphore)

option(vkexec_ENABLE_SAMPLE_SMOKE_TESTS "Register Vulkan sample runtime smoke tests" OFF)
if(vkexec_ENABLE_SAMPLE_SMOKE_TESTS)
  if(NOT BUILD_TESTING)
    message(FATAL_ERROR "vkexec_ENABLE_SAMPLE_SMOKE_TESTS requires BUILD_TESTING=ON")
  endif()

  function(vkexec_add_sample_smoke name target)
    add_test(NAME "vkexec.samples.${name}" COMMAND ${target} ${ARGN})
    set_tests_properties(
      "vkexec.samples.${name}"
      PROPERTIES LABELS
                 sample-smoke
                 TIMEOUT
                 180
                 FAIL_REGULAR_EXPRESSION
                 "VUID-|Validation Error|SYNC-HAZARD")
  endfunction()

  set(vkexec_SAMPLE_ASYNC_EXPECTED_QUEUE
      "shared"
      CACHE STRING "Expected async compute queue in sample smoke tests (shared or dedicated)")
  set_property(CACHE vkexec_SAMPLE_ASYNC_EXPECTED_QUEUE PROPERTY STRINGS shared dedicated)
  if(NOT
     vkexec_SAMPLE_ASYNC_EXPECTED_QUEUE
     MATCHES
     "^(shared|dedicated)$")
    message(FATAL_ERROR "vkexec_SAMPLE_ASYNC_EXPECTED_QUEUE must be shared or dedicated")
  endif()

  vkexec_add_sample_smoke(triangle vkexec_vs_triangle --frames=10)
  vkexec_add_sample_smoke(compute_nbody vkexec_vs_compute_nbody --frames=10)
  vkexec_add_sample_smoke(
    pipeline_barriers.fragment
    vkexec_vs_pipeline_barriers
    --frames=10
    --smoke-uncompressed
    --barrier=fragment)
  vkexec_add_sample_smoke(
    pipeline_barriers.conservative
    vkexec_vs_pipeline_barriers
    --frames=10
    --smoke-uncompressed
    --barrier=conservative)
  vkexec_add_sample_smoke(
    async_compute.single
    vkexec_vs_async_compute
    --frames=10
    --smoke-uncompressed
    --queues=single
    --expect-queue=shared)
  vkexec_add_sample_smoke(
    async_compute.async
    vkexec_vs_async_compute
    --frames=10
    --smoke-uncompressed
    --queues=async
    "--expect-queue=${vkexec_SAMPLE_ASYNC_EXPECTED_QUEUE}")
  vkexec_add_sample_smoke(timeline_semaphore vkexec_vs_timeline_semaphore --frames=10)

  foreach(
    target IN
    ITEMS vkexec_vs_triangle
          vkexec_vs_compute_nbody
          vkexec_vs_pipeline_barriers
          vkexec_vs_async_compute
          vkexec_vs_timeline_semaphore)
    foreach(
      argument IN
      ITEMS --frames=0
            --frames=-1
            --frames=abc
            --frames=+1
            --frames=1.5
            --frames=18446744073709551616
            --unknown)
      string(
        REPLACE "="
                "_"
                suffix
                "${argument}")
      string(
        REPLACE "--"
                ""
                suffix
                "${suffix}")
      add_test(NAME "vkexec.samples.cli.${target}.${suffix}" COMMAND ${target} "${argument}")
      set_tests_properties(
        "vkexec.samples.cli.${target}.${suffix}"
        PROPERTIES WILL_FAIL
                   TRUE
                   TIMEOUT
                   10
                   LABELS
                   sample-cli)
    endforeach()
  endforeach()

  foreach(target IN ITEMS vkexec_vs_pipeline_barriers vkexec_vs_async_compute)
    add_test(NAME "vkexec.samples.cli.${target}.uncompressed_without_frames" COMMAND ${target} --smoke-uncompressed)
    set_tests_properties(
      "vkexec.samples.cli.${target}.uncompressed_without_frames"
      PROPERTIES WILL_FAIL
                 TRUE
                 TIMEOUT
                 10
                 LABELS
                 sample-cli)
  endforeach()

  foreach(frame_count IN ITEMS 1 2)
    vkexec_add_sample_smoke("triangle.${frame_count}" vkexec_vs_triangle "--frames=${frame_count}")
    vkexec_add_sample_smoke("compute_nbody.${frame_count}" vkexec_vs_compute_nbody "--frames=${frame_count}")
    vkexec_add_sample_smoke(
      "pipeline_barriers.fragment.${frame_count}"
      vkexec_vs_pipeline_barriers
      "--frames=${frame_count}"
      --smoke-uncompressed
      --barrier=fragment)
    vkexec_add_sample_smoke(
      "async_compute.single.${frame_count}"
      vkexec_vs_async_compute
      "--frames=${frame_count}"
      --smoke-uncompressed
      --queues=single
      --expect-queue=shared)
    vkexec_add_sample_smoke(
      "async_compute.async.${frame_count}"
      vkexec_vs_async_compute
      "--frames=${frame_count}"
      --smoke-uncompressed
      --queues=async
      "--expect-queue=${vkexec_SAMPLE_ASYNC_EXPECTED_QUEUE}")
    vkexec_add_sample_smoke("timeline_semaphore.${frame_count}" vkexec_vs_timeline_semaphore "--frames=${frame_count}")
    foreach(
      name IN
      ITEMS triangle
            compute_nbody
            pipeline_barriers.fragment
            async_compute.single
            async_compute.async
            timeline_semaphore)
      set_property(
        TEST "vkexec.samples.${name}.${frame_count}"
        APPEND
        PROPERTY LABELS sample-extended)
    endforeach()
  endforeach()
endif()
