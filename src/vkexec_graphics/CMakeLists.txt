add_library(vkexec_graphics presenter.cpp swapchain.cpp graphics.cpp)

add_library(vkexec::vkexec_graphics ALIAS vkexec_graphics)
set_target_properties(vkexec_graphics PROPERTIES EXPORT_NAME vkexec_graphics)

target_link_libraries(
  vkexec_graphics
  PUBLIC vkexec::vkexec
  PRIVATE "$<BUILD_LOCAL_INTERFACE:vkexec_warnings>" vk-bootstrap::vk-bootstrap)

target_include_directories(
  vkexec_graphics ${WARNING_GUARD}
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)

target_compile_features(vkexec_graphics PUBLIC cxx_std_20)

vkexec_configure_linker(vkexec_graphics)

set_target_properties(
  vkexec_graphics
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_EXTENSIONS OFF
             CXX_VISIBILITY_PRESET default
             VISIBILITY_INLINES_HIDDEN YES)
