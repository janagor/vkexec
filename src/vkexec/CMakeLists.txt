add_library(
  vkexec
  context.cpp
  completion_waiter.cpp
  host_agent.cpp
  pipeline_cache.cpp
  compute_pipeline.cpp
  gpu_buffer.cpp
  timeline_semaphore.cpp
  descriptor_heap.cpp
  image.cpp
  image_view.cpp
  rendering.cpp
  sampler.cpp
  vma.cpp)

add_library(vkexec::vkexec ALIAS vkexec)

target_link_libraries(
  vkexec
  PUBLIC vkexec_options
         vkexec_warnings
         vkexec::vkexec_edsl
         cx_system_error::cx_system_error
         Vulkan::Vulkan
         STDEXEC::stdexec
         vk-bootstrap::vk-bootstrap
         GPUOpen::VulkanMemoryAllocator
  PRIVATE glslang::glslang glslang::SPIRV glslang::glslang-default-resource-limits)

target_include_directories(
  vkexec ${WARNING_GUARD}
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:include>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)

target_compile_features(vkexec PUBLIC cxx_std_${CMAKE_CXX_STANDARD})

vkexec_configure_linker(vkexec)

set_target_properties(
  vkexec
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_VISIBILITY_PRESET default
             VISIBILITY_INLINES_HIDDEN YES)
