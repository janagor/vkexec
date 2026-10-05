add_executable(vkexec_vs_async_compute main.cpp "${PROJECT_SOURCE_DIR}/examples/load_gltf_mesh.cpp")
target_link_libraries(
  vkexec_vs_async_compute
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::vkexec_graphics
          vkexec::vma
          vkexec_examples_glfw
          tinygltf::tinygltf
          vkexec_exception_options)
target_include_directories(vkexec_vs_async_compute PRIVATE "${PROJECT_SOURCE_DIR}/examples"
                                                            "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_definitions(vkexec_vs_async_compute
                           PRIVATE "VKEXEC_SAMPLE_ASSET_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/assets\"")
set_target_properties(vkexec_vs_async_compute PROPERTIES CXX_SCAN_FOR_MODULES OFF)
if(vkexec_BUILD_TOOLS)
  target_link_libraries(vkexec_vs_async_compute PRIVATE vkexec::tools)
endif()
vkexec_add_sample_shaders(
  vkexec_vs_async_compute
  forward.vert
  forward.frag
  shadow.vert
  shadow.frag
  composite.vert
  composite.frag
  threshold.comp
  blur_down.comp
  blur_up.comp)
vkexec_configure_linker(vkexec_vs_async_compute)
