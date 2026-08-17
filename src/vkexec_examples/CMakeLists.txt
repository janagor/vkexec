add_executable(compute compute.cpp)
add_executable(sort sort.cpp)
add_executable(passes passes.cpp)
add_executable(spirv spirv.cpp)
add_executable(triangle triangle.cpp)
add_executable(particles particles.cpp)
add_executable(layers layers.cpp)

target_link_libraries(compute PRIVATE vkexec::vkexec_options vkexec::vkexec_warnings vkexec::vkexec)

target_link_libraries(sort PRIVATE vkexec::vkexec_options vkexec::vkexec_warnings vkexec::vkexec)

target_link_libraries(passes PRIVATE vkexec::vkexec_options vkexec::vkexec_warnings vkexec::vkexec)

target_link_libraries(spirv PRIVATE vkexec::vkexec_options vkexec::vkexec_warnings vkexec::vkexec)

target_link_libraries(
  triangle
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::vkexec_graphics)

target_link_libraries(
  particles
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::vkexec_graphics)

target_link_libraries(
  layers
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::vkexec_graphics)

vkexec_configure_linker(compute)
vkexec_configure_linker(sort)
vkexec_configure_linker(passes)
vkexec_configure_linker(spirv)
vkexec_configure_linker(triangle)
vkexec_configure_linker(particles)
vkexec_configure_linker(layers)
