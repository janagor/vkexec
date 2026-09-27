cmake_minimum_required(VERSION 3.29)

project(vkexec_install_consumer LANGUAGES CXX)

find_package(vkexec CONFIG REQUIRED)

get_target_property(vkexec_compile_features vkexec::vkexec INTERFACE_COMPILE_FEATURES)
if(NOT
   "cxx_std_20"
   IN_LIST
   vkexec_compile_features)
  message(FATAL_ERROR "vkexec::vkexec does not export the cxx_std_20 requirement")
endif()

add_executable(vkexec_install_consumer main.cpp)
set_target_properties(vkexec_install_consumer PROPERTIES CXX_SCAN_FOR_MODULES OFF)
target_link_libraries(vkexec_install_consumer PRIVATE vkexec::vkexec)
