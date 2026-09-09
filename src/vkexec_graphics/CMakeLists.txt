add_library(
  vkexec_graphics
  window.cpp
  swapchain.cpp
  frame_present.cpp
  graphics.cpp
  mesh.cpp)

add_library(vkexec::vkexec_graphics ALIAS vkexec_graphics)

target_link_libraries(
  vkexec_graphics
  PUBLIC vkexec_options
         vkexec_warnings
         vkexec::vkexec
         vkexec::ext_timeline_semaphore
         glfw
  PRIVATE vk-bootstrap::vk-bootstrap)

target_include_directories(
  vkexec_graphics ${WARNING_GUARD} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include> $<INSTALL_INTERFACE:include>)

target_compile_features(vkexec_graphics PUBLIC cxx_std_${CMAKE_CXX_STANDARD})

vkexec_configure_linker(vkexec_graphics)

set_target_properties(
  vkexec_graphics
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_VISIBILITY_PRESET default
             VISIBILITY_INLINES_HIDDEN YES)
