add_library(vkexec_ext_descriptor_heap
  descriptor_heap/descriptor_heap.cpp
  descriptor_heap/extension.cpp
  descriptor_heap/procs.cpp
  descriptor_heap/buffer.cpp
  descriptor_heap/push_data.cpp
  descriptor_heap/pass.cpp
  descriptor_heap/compute_pipeline.cpp
  descriptor_heap/heap_compute_pipeline.cpp)
add_library(vkexec::ext_descriptor_heap ALIAS vkexec_ext_descriptor_heap)

add_library(vkexec_ext_dynamic_rendering
  dynamic_rendering/extension.cpp dynamic_rendering/rendering.cpp)
add_library(vkexec::ext_dynamic_rendering ALIAS vkexec_ext_dynamic_rendering)

target_include_directories(
  vkexec_ext_descriptor_heap
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

target_link_libraries(vkexec_ext_descriptor_heap PUBLIC Vulkan::Vulkan vkexec::vkexec)

target_compile_features(vkexec_ext_descriptor_heap PUBLIC cxx_std_${CMAKE_CXX_STANDARD})

vkexec_configure_linker(vkexec_ext_descriptor_heap)

set_target_properties(
  vkexec_ext_descriptor_heap
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_VISIBILITY_PRESET default
             VISIBILITY_INLINES_HIDDEN YES)

target_include_directories(
  vkexec_ext_dynamic_rendering
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

target_link_libraries(vkexec_ext_dynamic_rendering PUBLIC Vulkan::Vulkan vkexec::vkexec)

target_compile_features(vkexec_ext_dynamic_rendering PUBLIC cxx_std_${CMAKE_CXX_STANDARD})

vkexec_configure_linker(vkexec_ext_dynamic_rendering)

set_target_properties(
  vkexec_ext_dynamic_rendering
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_VISIBILITY_PRESET default
             VISIBILITY_INLINES_HIDDEN YES)
