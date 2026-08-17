add_executable(vkexec_example compute.cpp)
add_executable(vkexec_sort_example sort.cpp)
add_executable(vkexec_triangle_example triangle.cpp)
add_executable(vkexec_particles_example particles.cpp)
add_executable(vkexec_layers_example layers.cpp)

target_link_libraries(
  vkexec_example
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec)

target_link_libraries(
  vkexec_sort_example
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec)

target_link_libraries(
  vkexec_triangle_example
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec)

target_link_libraries(
  vkexec_particles_example
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec)

target_link_libraries(
  vkexec_layers_example
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec)

vkexec_configure_linker(vkexec_example)
vkexec_configure_linker(vkexec_sort_example)
vkexec_configure_linker(vkexec_triangle_example)
vkexec_configure_linker(vkexec_particles_example)
vkexec_configure_linker(vkexec_layers_example)
