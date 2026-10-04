find_program(
  VKEXEC_GLSLC
  NAMES glslc
  HINTS "$ENV{VULKAN_SDK}/bin")

function(vkexec_add_sample_shaders target)
  if(NOT VKEXEC_GLSLC)
    message(FATAL_ERROR "glslc is required to build the Vulkan Samples examples")
  endif()

  set(shader_outputs)
  foreach(shader IN LISTS ARGN)
    get_filename_component(shader_name "${shader}" NAME)
    set(output "${CMAKE_CURRENT_BINARY_DIR}/shaders/${shader_name}.spv")
    add_custom_command(
      OUTPUT "${output}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
      COMMAND "${VKEXEC_GLSLC}" --target-env=vulkan1.3 "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}" -o "${output}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}"
      VERBATIM)
    list(APPEND shader_outputs "${output}")
  endforeach()
  add_custom_target(${target}_shaders DEPENDS ${shader_outputs})
  add_dependencies(${target} ${target}_shaders)
  target_compile_definitions(${target} PRIVATE "VKEXEC_SAMPLE_SHADER_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/shaders\"")
endfunction()

add_subdirectory(hello_triangle_1_3)
add_subdirectory(compute_nbody)
add_subdirectory(pipeline_barriers)
