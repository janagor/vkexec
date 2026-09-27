cmake_minimum_required(VERSION 3.29)

project(vkexec_subdirectory_consumer LANGUAGES CXX)

# Configure this fixture with 17 and 23 to check both sides of the minimum.
if(NOT
   CMAKE_CXX_STANDARD
   MATCHES
   "^(17|23)$")
  message(FATAL_ERROR "Configure this fixture with CMAKE_CXX_STANDARD=17 or 23")
endif()

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." vkexec)

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
