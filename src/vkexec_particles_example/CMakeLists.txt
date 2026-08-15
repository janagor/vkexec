add_executable(vkexec_particles_example main.cpp)

target_link_libraries(
  vkexec_particles_example
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec)

vkexec_configure_linker(vkexec_particles_example)
