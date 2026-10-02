add_library(
  vkexec
  error.cpp
  error_helpers.cpp
  context.cpp
  submission.cpp
  completion_waiter.cpp
  host_agent.cpp
  bind_resources.cpp
  compute_pipeline.cpp
  push.cpp
  barrier.cpp
  image_view.cpp
  submit_scope.cpp
  pass.cpp
  vulkan_requirements.cpp
  sampler.cpp)

add_library(vkexec::vkexec ALIAS vkexec)
set_target_properties(vkexec PROPERTIES EXPORT_NAME vkexec)

target_link_libraries(
  vkexec
  PUBLIC "$<BUILD_LOCAL_INTERFACE:vkexec_vulkan>"
         "$<INSTALL_INTERFACE:Vulkan::Headers>"
         "$<INSTALL_INTERFACE:$<LINK_ONLY:Vulkan::Vulkan>>"
         STDEXEC::stdexec
  PRIVATE vk-bootstrap::vk-bootstrap "$<BUILD_LOCAL_INTERFACE:vkexec_warnings>")

target_include_directories(
  vkexec ${WARNING_GUARD}
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)

target_compile_features(vkexec PUBLIC cxx_std_20)

vkexec_configure_linker(vkexec)

set_target_properties(
  vkexec
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_EXTENSIONS OFF
             CXX_VISIBILITY_PRESET default
             VISIBILITY_INLINES_HIDDEN YES)
