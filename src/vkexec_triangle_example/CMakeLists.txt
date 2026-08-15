add_executable(vkexec_triangle_example main.cpp)

target_link_libraries(
  vkexec_triangle_example
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec)

vkexec_configure_linker(vkexec_triangle_example)
