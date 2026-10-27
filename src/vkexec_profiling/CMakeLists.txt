add_library(vkexec_profiling profiler.cpp manual_capture.cpp chrome_trace.cpp)
add_library(vkexec::profiling ALIAS vkexec_profiling)
set_target_properties(vkexec_profiling PROPERTIES EXPORT_NAME profiling)

target_link_libraries(vkexec_profiling PUBLIC vkexec::vkexec PRIVATE "$<BUILD_LOCAL_INTERFACE:vkexec_warnings>")
target_include_directories(
  vkexec_profiling ${WARNING_GUARD}
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(vkexec_profiling PUBLIC cxx_std_20)
vkexec_configure_linker(vkexec_profiling)
set_target_properties(
  vkexec_profiling
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_EXTENSIONS OFF
             CXX_VISIBILITY_PRESET default
             VISIBILITY_INLINES_HIDDEN YES)
