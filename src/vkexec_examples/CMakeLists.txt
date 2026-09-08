add_executable(spirv spirv.cpp)
add_executable(triangle triangle.cpp)
add_executable(heap_present heap_present.cpp)

target_link_libraries(spirv PRIVATE vkexec::vkexec_options vkexec::vkexec_warnings vkexec::vkexec)

target_link_libraries(
  triangle
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::vkexec_graphics)

target_link_libraries(
  heap_present
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::vkexec_graphics)

vkexec_configure_linker(spirv)
vkexec_configure_linker(triangle)
vkexec_configure_linker(heap_present)
