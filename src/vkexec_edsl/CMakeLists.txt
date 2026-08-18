add_library(vkexec_edsl spirv.cpp types.cpp trace.cpp)

add_library(vkexec::vkexec_edsl ALIAS vkexec_edsl)

target_link_libraries(
  vkexec_edsl
  PUBLIC vkexec_options
         vkexec_warnings
         Boost::describe
         Boost::mp11
  PRIVATE glslang::glslang glslang::SPIRV glslang::glslang-default-resource-limits)

target_include_directories(
  vkexec_edsl ${WARNING_GUARD}
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:include>
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)

target_compile_features(vkexec_edsl PUBLIC cxx_std_${CMAKE_CXX_STANDARD})

vkexec_configure_linker(vkexec_edsl)

set_target_properties(
  vkexec_edsl
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_VISIBILITY_PRESET default
             VISIBILITY_INLINES_HIDDEN YES)
