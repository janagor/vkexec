add_executable(dynamic_rendering dynamic_rendering.cpp)

target_link_libraries(
  dynamic_rendering
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::ext_dynamic_rendering
          vkexec::vkexec_graphics)

vkexec_configure_linker(dynamic_rendering)
