if(NOT DEFINED VKEXEC_SOURCE_DIR)
  message(FATAL_ERROR "VKEXEC_SOURCE_DIR is required")
endif()

file(
  GLOB_RECURSE
  _vkexec_core_sources
  "${VKEXEC_SOURCE_DIR}/include/vkexec/*.hpp"
  "${VKEXEC_SOURCE_DIR}/src/vkexec/*.cpp"
  "${VKEXEC_SOURCE_DIR}/include/vkexec_graphics/*.hpp"
  "${VKEXEC_SOURCE_DIR}/src/vkexec_graphics/*.cpp"
  "${VKEXEC_SOURCE_DIR}/include/vkexec_extensions/*.hpp"
  "${VKEXEC_SOURCE_DIR}/src/vkexec_extensions/*.cpp")
foreach(_vkexec_source IN LISTS _vkexec_core_sources)
  file(READ "${_vkexec_source}" _vkexec_contents)
  if(_vkexec_contents MATCHES "vk_mem_alloc|VmaAllocator|VmaAllocation|vma[A-Z]")
    message(FATAL_ERROR "VMA implementation leaked into core: ${_vkexec_source}")
  endif()
endforeach()
