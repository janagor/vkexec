add_library(vkexec context.cpp spirv.cpp pipeline_cache.cpp window.cpp)

add_library(vkexec::vkexec ALIAS vkexec)

target_link_libraries(
  vkexec
  PUBLIC vkexec_options
         vkexec_warnings
         Vulkan::Vulkan
         STDEXEC::stdexec
         glfw
  PRIVATE glslang::glslang
          glslang::SPIRV
          glslang::glslang-default-resource-limits)

target_include_directories(
  vkexec ${WARNING_GUARD} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                                  $<INSTALL_INTERFACE:include>)

target_compile_features(vkexec PUBLIC cxx_std_${CMAKE_CXX_STANDARD})

vkexec_configure_linker(vkexec)

set_target_properties(
  vkexec
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_VISIBILITY_PRESET default
             VISIBILITY_INLINES_HIDDEN YES)
