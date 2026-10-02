cmake_minimum_required(VERSION 3.29)

project(vkexec_subdirectory_consumer LANGUAGES CXX)

# Configure this fixture with 17 and 23 to check both sides of the minimum.
if(NOT
   CMAKE_CXX_STANDARD
   MATCHES
   "^(17|23)$")
  message(FATAL_ERROR "Configure this fixture with CMAKE_CXX_STANDARD=17 or 23")
endif()

# Supply aliased Vulkan targets whose usage requirements belong to the parent.
if(NOT Vulkan_LIBRARY OR NOT Vulkan_INCLUDE_DIR)
  message(FATAL_ERROR "Vulkan_LIBRARY and Vulkan_INCLUDE_DIR are required for the parent Vulkan fixture")
endif()

add_library(
  parent_vulkan_loader
  INTERFACE
  IMPORTED
  GLOBAL)
set_target_properties(
  parent_vulkan_loader
  PROPERTIES INTERFACE_LINK_LIBRARIES "${Vulkan_LIBRARY}"
             INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIR}"
             INTERFACE_COMPILE_DEFINITIONS "VKEXEC_PARENT_VULKAN_LOADER=1")
add_library(Vulkan::Vulkan ALIAS parent_vulkan_loader)

add_library(
  parent_vulkan_headers
  INTERFACE
  IMPORTED
  GLOBAL)
set_target_properties(parent_vulkan_headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIR}"
                                                       INTERFACE_COMPILE_DEFINITIONS "VKEXEC_PARENT_VULKAN_HEADERS=1")
add_library(Vulkan::Headers ALIAS parent_vulkan_headers)

foreach(target IN ITEMS parent_vulkan_loader parent_vulkan_headers)
  foreach(property IN ITEMS INTERFACE_INCLUDE_DIRECTORIES INTERFACE_COMPILE_DEFINITIONS INTERFACE_LINK_LIBRARIES)
    get_property(
      parent_vulkan_${target}_${property}_was_set
      TARGET ${target}
      PROPERTY ${property}
      SET)
    get_property(
      parent_vulkan_${target}_${property}_before
      TARGET ${target}
      PROPERTY ${property})
  endforeach()
endforeach()

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." vkexec)

foreach(target IN ITEMS parent_vulkan_loader parent_vulkan_headers)
  foreach(property IN ITEMS INTERFACE_INCLUDE_DIRECTORIES INTERFACE_COMPILE_DEFINITIONS INTERFACE_LINK_LIBRARIES)
    get_property(
      parent_vulkan_${target}_${property}_is_set
      TARGET ${target}
      PROPERTY ${property}
      SET)
    get_property(
      parent_vulkan_${target}_${property}_after
      TARGET ${target}
      PROPERTY ${property})
    if(NOT
       "${parent_vulkan_${target}_${property}_is_set}"
       STREQUAL
       "${parent_vulkan_${target}_${property}_was_set}"
       OR NOT
          "${parent_vulkan_${target}_${property}_after}"
          STREQUAL
          "${parent_vulkan_${target}_${property}_before}")
      message(FATAL_ERROR "vkexec modified parent-owned ${target} ${property}")
    endif()
  endforeach()
endforeach()

get_target_property(vkexec_vulkan_links vkexec_vulkan INTERFACE_LINK_LIBRARIES)
if(NOT
   vkexec_vulkan_links
   STREQUAL
   "Vulkan::Headers;$<LINK_ONLY:Vulkan::Vulkan>")
  message(FATAL_ERROR "vkexec_vulkan must use parent Vulkan headers and a link-only loader")
endif()

get_target_property(vkexec_compile_features vkexec::vkexec INTERFACE_COMPILE_FEATURES)
if(NOT
   "cxx_std_20"
   IN_LIST
   vkexec_compile_features)
  message(FATAL_ERROR "vkexec::vkexec must export cxx_std_20")
endif()

add_library(vkexec_subdirectory_consumer OBJECT main.cpp)
set_target_properties(vkexec_subdirectory_consumer PROPERTIES CXX_SCAN_FOR_MODULES OFF)
target_link_libraries(vkexec_subdirectory_consumer PRIVATE vkexec::vkexec)

add_executable(vkexec_unrelated unrelated.cpp)
get_target_property(unrelated_standard vkexec_unrelated CXX_STANDARD)
if(NOT
   unrelated_standard
   STREQUAL
   "${CMAKE_CXX_STANDARD}")
  message(FATAL_ERROR "vkexec changed the parent-selected CXX_STANDARD")
endif()
