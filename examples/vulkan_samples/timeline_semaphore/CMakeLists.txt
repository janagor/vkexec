add_executable(vkexec_vs_timeline_semaphore main.cpp)
target_link_libraries(
  vkexec_vs_timeline_semaphore
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::vkexec_graphics
          vkexec::vma
          vkexec::ext_timeline_semaphore
          vkexec_examples_glfw
          vkexec_exception_options)
target_include_directories(vkexec_vs_timeline_semaphore PRIVATE "${PROJECT_SOURCE_DIR}/examples"
                                                                "${CMAKE_CURRENT_SOURCE_DIR}/..")
set_target_properties(vkexec_vs_timeline_semaphore PROPERTIES CXX_SCAN_FOR_MODULES OFF)
vkexec_add_sample_shaders(
  vkexec_vs_timeline_semaphore
  game_of_life_init.comp
  game_of_life_update.comp
  game_of_life_mutate.comp
  render.vert
  render.frag)
vkexec_configure_linker(vkexec_vs_timeline_semaphore)
