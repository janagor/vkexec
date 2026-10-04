add_executable(vkexec_vs_triangle main.cpp)
target_link_libraries(
  vkexec_vs_triangle
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::vkexec_graphics
          vkexec::vma
          vkexec_examples_glfw
          vkexec_exception_options)
target_include_directories(vkexec_vs_triangle PRIVATE
  "${PROJECT_SOURCE_DIR}/examples"
  "${CMAKE_CURRENT_SOURCE_DIR}/..")
set_target_properties(vkexec_vs_triangle PROPERTIES CXX_SCAN_FOR_MODULES OFF)
if(vkexec_BUILD_TOOLS)
  target_link_libraries(vkexec_vs_triangle PRIVATE vkexec::tools)
endif()
vkexec_add_sample_shaders(vkexec_vs_triangle triangle.vert triangle.frag)
vkexec_configure_linker(vkexec_vs_triangle)
