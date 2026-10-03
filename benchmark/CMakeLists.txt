add_executable(vkexec_benchmarks pass_graph.cpp allocation_counter.cpp)
set_target_properties(
  vkexec_benchmarks
  PROPERTIES CXX_SCAN_FOR_MODULES OFF
             CXX_CLANG_TIDY ""
             CXX_CPPCHECK "")
target_link_libraries(vkexec_benchmarks PRIVATE vkexec::vkexec benchmark::benchmark benchmark::benchmark_main)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 22)
  target_compile_options(vkexec_benchmarks PRIVATE -Wno-c2y-extensions)
endif()
vkexec_configure_linker(vkexec_benchmarks)
