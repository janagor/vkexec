cmake_minimum_required(VERSION 3.29)

project(vkexec_install_consumer LANGUAGES CXX)

find_package(vkexec CONFIG REQUIRED)
cmake_path(
  IS_PREFIX
  VKEXEC_PREFIX
  "${vkexec_DIR}"
  NORMALIZE
  vkexec_from_test_prefix)
if(NOT vkexec_from_test_prefix)
  message(FATAL_ERROR "vkexec was found outside the relocated test prefix: ${vkexec_DIR}")
endif()

set(vkexec_required_targets
    vkexec::vkexec
    vkexec::features
    vkexec::vkexec_graphics
    vkexec::ext_descriptor_heap
    vkexec::ext_dynamic_rendering
    vkexec::ext_timeline_semaphore)
if(VKEXEC_EXPECT_VMA)
  list(APPEND vkexec_required_targets vkexec::vma)
elseif(TARGET vkexec::vma)
  message(FATAL_ERROR "Unexpected optional target: vkexec::vma")
endif()
if(VKEXEC_EXPECT_TOOLS)
  list(APPEND vkexec_required_targets vkexec::tools)
elseif(TARGET vkexec::tools)
  message(FATAL_ERROR "Unexpected optional target: vkexec::tools")
endif()
foreach(target IN LISTS vkexec_required_targets)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "Missing installed target: ${target}")
  endif()
  get_target_property(include_directories ${target} INTERFACE_INCLUDE_DIRECTORIES)
  foreach(include_directory IN LISTS include_directories)
    string(FIND "${include_directory}" "${VKEXEC_PREFIX}/" prefix_position)
    if(NOT
       prefix_position
       EQUAL
       -1)
      continue()
    endif()
    foreach(
      forbidden IN
      ITEMS "${VKEXEC_FORBIDDEN_SOURCE}"
            "${VKEXEC_FORBIDDEN_BUILD}"
            "/_deps/"
            "CPM")
      string(FIND "${include_directory}" "${forbidden}" leak_position)
      if(NOT
         leak_position
         EQUAL
         -1)
        message(FATAL_ERROR "Build path leaked from ${target}: ${include_directory}")
      endif()
    endforeach()
  endforeach()
  get_target_property(link_libraries ${target} INTERFACE_LINK_LIBRARIES)
  foreach(link_library IN LISTS link_libraries)
    foreach(
      forbidden IN
      ITEMS "${VKEXEC_FORBIDDEN_SOURCE}"
            "${VKEXEC_FORBIDDEN_BUILD}"
            "/_deps/"
            "CPM")
      string(FIND "${link_library}" "${forbidden}" leak_position)
      if(NOT
         leak_position
         EQUAL
         -1)
        message(FATAL_ERROR "Build path leaked from ${target}: ${link_library}")
      endif()
    endforeach()
  endforeach()
endforeach()
foreach(
  target IN
  ITEMS vkexec::vkexec_options
        vkexec::vkexec_warnings
        vkexec::stdexec
        vkexec::vk-bootstrap
        vkexec::VulkanMemoryAllocator
        vkexec::glslang)
  if(TARGET ${target})
    message(FATAL_ERROR "Internal or third-party target leaked: ${target}")
  endif()
endforeach()

get_target_property(vkexec_compile_features vkexec::vkexec INTERFACE_COMPILE_FEATURES)
if(NOT
   "cxx_std_20"
   IN_LIST
   vkexec_compile_features)
  message(FATAL_ERROR "vkexec::vkexec does not export the cxx_std_20 requirement")
endif()

function(vkexec_add_install_consumer name target)
  add_executable(consume_${name} ${name}.cpp)
  set_target_properties(consume_${name} PROPERTIES CXX_SCAN_FOR_MODULES OFF)
  target_link_libraries(consume_${name} PRIVATE ${target})
endfunction()

vkexec_add_install_consumer(core vkexec::vkexec)
vkexec_add_install_consumer(features vkexec::features)
vkexec_add_install_consumer(graphics vkexec::vkexec_graphics)
vkexec_add_install_consumer(ext_descriptor_heap vkexec::ext_descriptor_heap)
vkexec_add_install_consumer(ext_dynamic_rendering vkexec::ext_dynamic_rendering)
vkexec_add_install_consumer(ext_timeline_semaphore vkexec::ext_timeline_semaphore)
if(VKEXEC_EXPECT_VMA)
  vkexec_add_install_consumer(vma vkexec::vma)
endif()
if(VKEXEC_EXPECT_TOOLS)
  vkexec_add_install_consumer(tools vkexec::tools)
endif()
add_executable(consume_all main.cpp)
set_target_properties(consume_all PROPERTIES CXX_SCAN_FOR_MODULES OFF)
target_link_libraries(consume_all PRIVATE ${vkexec_required_targets})
