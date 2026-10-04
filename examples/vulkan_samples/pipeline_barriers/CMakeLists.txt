add_executable(vkexec_vs_pipeline_barriers main.cpp "${PROJECT_SOURCE_DIR}/examples/load_gltf_mesh.cpp")
target_link_libraries(
  vkexec_vs_pipeline_barriers
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::vkexec_graphics
          vkexec::features
          vkexec::vma
          vkexec_examples_glfw
          tinygltf::tinygltf
          vkexec_exception_options)
target_include_directories(vkexec_vs_pipeline_barriers PRIVATE "${PROJECT_SOURCE_DIR}/examples"
                                                               "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_definitions(vkexec_vs_pipeline_barriers
                           PRIVATE "VKEXEC_SAMPLE_ASSET_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/assets\"")
set_target_properties(vkexec_vs_pipeline_barriers PROPERTIES CXX_SCAN_FOR_MODULES OFF)
if(vkexec_BUILD_TOOLS)
  target_link_libraries(vkexec_vs_pipeline_barriers PRIVATE vkexec::tools)
endif()
vkexec_add_sample_shaders(
  vkexec_vs_pipeline_barriers
  geometry.vert
  geometry.frag
  lighting.vert
  lighting.frag
  albedo.frag)
vkexec_configure_linker(vkexec_vs_pipeline_barriers)
