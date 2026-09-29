add_library(
  vkexec_features
  query.cpp
  timeline_semaphore.cpp
  buffer_device_address.cpp
  dynamic_rendering.cpp
  synchronization2.cpp)
add_library(vkexec::features ALIAS vkexec_features)
set_target_properties(vkexec_features PROPERTIES EXPORT_NAME features)

target_include_directories(
  vkexec_features
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)

target_link_libraries(
  vkexec_features
  PUBLIC vkexec::vkexec Vulkan::Vulkan
  PRIVATE vk-bootstrap::vk-bootstrap "$<BUILD_LOCAL_INTERFACE:vkexec_warnings>")

target_compile_features(vkexec_features PUBLIC cxx_std_20)

vkexec_configure_linker(vkexec_features)

set_target_properties(
  vkexec_features
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_EXTENSIONS OFF
             CXX_VISIBILITY_PRESET default
             VISIBILITY_INLINES_HIDDEN YES)
