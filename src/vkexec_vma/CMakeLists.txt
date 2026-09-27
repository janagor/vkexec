add_library(
  vkexec_vma
  allocator.cpp
  gpu_buffer.cpp
  image.cpp
  copy.cpp
  vma.cpp)

add_library(vkexec::vma ALIAS vkexec_vma)

target_link_libraries(
  vkexec_vma
  PUBLIC vkexec::vkexec GPUOpen::VulkanMemoryAllocator
  PRIVATE vkexec_warnings)

target_include_directories(vkexec_vma ${WARNING_GUARD} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                                              $<INSTALL_INTERFACE:include>)

target_compile_features(vkexec_vma PUBLIC cxx_std_20)
target_compile_definitions(vkexec_vma PUBLIC VKEXEC_ENABLE_EXCEPTIONS=$<BOOL:${VKEXEC_ENABLE_EXCEPTIONS}>)
vkexec_configure_linker(vkexec_vma)

set_target_properties(
  vkexec_vma
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_EXTENSIONS OFF
             CXX_VISIBILITY_PRESET default
             VISIBILITY_INLINES_HIDDEN YES)
