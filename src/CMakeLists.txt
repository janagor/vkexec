add_library(vkexec_private_headers INTERFACE)
target_include_directories(vkexec_private_headers INTERFACE "${PROJECT_SOURCE_DIR}/src")

add_subdirectory(vkexec)
add_subdirectory(vkexec_features)
add_subdirectory(vkexec_graphics)
if(vkexec_BUILD_VMA)
  add_subdirectory(vkexec_vma)
endif()
add_subdirectory(vkexec_extensions)
if(vkexec_BUILD_TOOLS)
  add_subdirectory(vkexec_tools)
endif()

set(vkexec_compiled_targets
    vkexec
    vkexec_features
    vkexec_graphics
    vkexec_ext_descriptor_heap
    vkexec_ext_dynamic_rendering
    vkexec_ext_timeline_semaphore)
if(vkexec_BUILD_VMA)
  list(APPEND vkexec_compiled_targets vkexec_vma)
endif()
if(vkexec_BUILD_TOOLS)
  list(APPEND vkexec_compiled_targets vkexec_tools)
endif()
foreach(vkexec_target IN LISTS vkexec_compiled_targets)
  target_link_libraries(${vkexec_target} PRIVATE "$<BUILD_LOCAL_INTERFACE:vkexec_exception_options>")
  if(vkexec_ENABLE_SANITIZER_THREAD)
    target_link_libraries(${vkexec_target} PRIVATE "$<BUILD_LOCAL_INTERFACE:vkexec_options>")
  endif()
endforeach()
