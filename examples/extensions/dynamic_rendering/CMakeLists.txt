add_executable(dynamic_rendering dynamic_rendering.cpp)
set_target_properties(dynamic_rendering PROPERTIES CXX_SCAN_FOR_MODULES OFF)

target_link_libraries(
  dynamic_rendering
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::ext_dynamic_rendering
          vkexec::vkexec_graphics
          vkexec_examples_glfw)

vkexec_configure_linker(dynamic_rendering)
