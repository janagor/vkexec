if(NOT DEFINED VKEXEC_SOURCE_DIR)
  message(FATAL_ERROR "VKEXEC_SOURCE_DIR is required")
endif()

file(GLOB_RECURSE _vkexec_public_headers
  "${VKEXEC_SOURCE_DIR}/include/*.hpp")
foreach(_vkexec_header IN LISTS _vkexec_public_headers)
  file(READ "${_vkexec_header}" _vkexec_contents)
  if(_vkexec_contents MATCHES "VkBootstrap|vkb::")
    message(FATAL_ERROR "vk-bootstrap leaked into public header: ${_vkexec_header}")
  endif()
endforeach()

file(GLOB_RECURSE _vkexec_implementation_sources
  "${VKEXEC_SOURCE_DIR}/src/*.cpp"
  "${VKEXEC_SOURCE_DIR}/src/*.hpp")
foreach(_vkexec_source IN LISTS _vkexec_implementation_sources)
  if(_vkexec_source MATCHES "/src/vkexec/context.cpp$" OR
     _vkexec_source MATCHES "/src/vkexec/detail/vk_bootstrap_[^/]+$" OR
     _vkexec_source MATCHES "/src/vkexec_graphics/swapchain.cpp$")
    continue()
  endif()
  file(READ "${_vkexec_source}" _vkexec_contents)
  if(_vkexec_contents MATCHES "VkBootstrap|vkb::")
    message(FATAL_ERROR "vk-bootstrap leaked outside core adapter: ${_vkexec_source}")
  endif()
endforeach()
