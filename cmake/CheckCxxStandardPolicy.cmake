if(NOT DEFINED VKEXEC_SOURCE_DIR)
  message(FATAL_ERROR "VKEXEC_SOURCE_DIR is required")
endif()

file(READ "${VKEXEC_SOURCE_DIR}/CMakeLists.txt" root_cmake)
if(root_cmake MATCHES "set\\([ \t\r\n]*CMAKE_CXX_(STANDARD|EXTENSIONS)"
   OR root_cmake MATCHES "target_compile_features\\([ \t\r\n]*vkexec_options")
  message(FATAL_ERROR "Root CMake must not set a global C++ dialect or put it on vkexec_options")
endif()

set(vkexec_targets
    vkexec
    vkexec_features
    vkexec_graphics
    vkexec_vma
    vkexec_tools
    vkexec_ext_descriptor_heap
    vkexec_ext_dynamic_rendering
    vkexec_ext_timeline_semaphore)

file(GLOB component_cmake "${VKEXEC_SOURCE_DIR}/src/*/CMakeLists.txt")
foreach(cmake_file IN LISTS component_cmake)
  file(READ "${cmake_file}" cmake_text)
  if(cmake_text MATCHES "cxx_std_\\$\\{CMAKE_CXX_STANDARD\\}" OR cmake_text MATCHES "cxx_std_23")
    message(FATAL_ERROR "Variable-driven or C++23 feature found in ${cmake_file}")
  endif()
endforeach()

foreach(vkexec_target IN LISTS vkexec_targets)
  set(found FALSE)
  foreach(cmake_file IN LISTS component_cmake)
    file(READ "${cmake_file}" cmake_text)
    if(cmake_text MATCHES "target_compile_features\\([ \t\r\n]*${vkexec_target}[ \t\r\n]+PUBLIC[ \t\r\n]+cxx_std_20\\)"
       AND cmake_text MATCHES
           "set_target_properties\\([ \t\r\n]*${vkexec_target}[ \t\r\n]+PROPERTIES[^)]*CXX_EXTENSIONS[ \t\r\n]+OFF")
      set(found TRUE)
      break()
    endif()
  endforeach()
  if(NOT found)
    message(FATAL_ERROR "${vkexec_target} must declare PUBLIC cxx_std_20 and CXX_EXTENSIONS OFF")
  endif()
endforeach()
