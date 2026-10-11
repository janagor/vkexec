add_executable(vkexec_resource_regions_benchmark resource_regions.cpp)
target_link_libraries(
  vkexec_resource_regions_benchmark
  PRIVATE vkexec::vkexec
          vkexec::vkexec_options
          vkexec::vkexec_warnings
          benchmark::benchmark
          benchmark::benchmark_main)
set_target_properties(vkexec_resource_regions_benchmark PROPERTIES CXX_SCAN_FOR_MODULES OFF)
