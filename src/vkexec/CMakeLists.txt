add_library(
  vkexec
  error.cpp
  error_helpers.cpp
  context.cpp
  completion_waiter.cpp
  host_agent.cpp
  bind_resources.cpp
  compute_pipeline.cpp
  push.cpp
  gpu_buffer.cpp
  barrier.cpp
  copy.cpp
  image.cpp
  image_view.cpp
  submit_scope.cpp
  pass.cpp
  vulkan_requirements.cpp
  sampler.cpp
  vma.cpp)

add_library(vkexec::vkexec ALIAS vkexec)

target_link_libraries(
  vkexec
  PUBLIC
    "$<BUILD_INTERFACE:$<1:Boost::system>;Vulkan::Headers;Vulkan::Vulkan;STDEXEC::stdexec;vk-bootstrap::vk-bootstrap;GPUOpen::VulkanMemoryAllocator>"
    "$<INSTALL_INTERFACE:$<1:Boost::headers>;Vulkan::Vulkan;STDEXEC::stdexec;vk-bootstrap::vk-bootstrap;GPUOpen::VulkanMemoryAllocator>"
  PRIVATE vkexec_warnings)

target_include_directories(
  vkexec ${WARNING_GUARD}
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:include>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)

target_compile_features(vkexec PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
target_compile_definitions(vkexec PUBLIC VKEXEC_ENABLE_EXCEPTIONS=$<BOOL:${VKEXEC_ENABLE_EXCEPTIONS}>)

vkexec_configure_linker(vkexec)

set_target_properties(
  vkexec
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_VISIBILITY_PRESET default
             VISIBILITY_INLINES_HIDDEN YES)
