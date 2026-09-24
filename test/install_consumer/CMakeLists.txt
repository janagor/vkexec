cmake_minimum_required(VERSION 3.29)

project(vkexec_install_consumer LANGUAGES CXX)

find_package(vkexec CONFIG REQUIRED)

add_executable(vkexec_install_consumer main.cpp)
set_target_properties(vkexec_install_consumer PROPERTIES CXX_SCAN_FOR_MODULES OFF)
target_link_libraries(vkexec_install_consumer PRIVATE vkexec::vkexec)
target_compile_features(vkexec_install_consumer PRIVATE cxx_std_20)
