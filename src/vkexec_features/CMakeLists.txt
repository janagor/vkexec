add_library(
  vkexec_features
  query.cpp
  timeline_semaphore.cpp
  buffer_device_address.cpp
  dynamic_rendering.cpp)
add_library(vkexec::features ALIAS vkexec_features)

target_include_directories(vkexec_features PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                                  $<INSTALL_INTERFACE:include>)

target_link_libraries(vkexec_features PUBLIC vkexec::vkexec Vulkan::Vulkan)

target_compile_features(vkexec_features PUBLIC cxx_std_${CMAKE_CXX_STANDARD})

vkexec_configure_linker(vkexec_features)

set_target_properties(
  vkexec_features
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_VISIBILITY_PRESET default
             VISIBILITY_INLINES_HIDDEN YES)
