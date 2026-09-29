add_library(vkexec_tools spirv_compile.cpp pipeline_compile.cpp pipeline_compile_heap.cpp)
add_library(vkexec::tools ALIAS vkexec_tools)
set_target_properties(vkexec_tools PROPERTIES EXPORT_NAME tools)

target_link_libraries(
  vkexec_tools
  PUBLIC vkexec::vkexec vkexec::vkexec_graphics vkexec::ext_descriptor_heap
  PRIVATE "$<BUILD_LOCAL_INTERFACE:vkexec_warnings>"
          glslang::glslang
          glslang::SPIRV
          glslang::glslang-default-resource-limits)
target_include_directories(
  vkexec_tools ${WARNING_GUARD}
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(vkexec_tools PUBLIC cxx_std_20)
vkexec_configure_linker(vkexec_tools)
set_target_properties(
  vkexec_tools
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_EXTENSIONS OFF
             CXX_VISIBILITY_PRESET default
             VISIBILITY_INLINES_HIDDEN YES)
