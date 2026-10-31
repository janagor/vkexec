add_executable(vkexec_resource_regions_benchmark resource_regions.cpp)
target_link_libraries(
  vkexec_resource_regions_benchmark
  PRIVATE vkexec::vkexec
          vkexec::vkexec_options
          vkexec::vkexec_warnings
          benchmark::benchmark
          benchmark::benchmark_main)
set_target_properties(vkexec_resource_regions_benchmark PROPERTIES CXX_SCAN_FOR_MODULES OFF)

add_executable(vkexec_graph_planning_benchmark graph_planning.cpp)
target_link_libraries(
  vkexec_graph_planning_benchmark
  PRIVATE vkexec::vkexec
          vkexec::vkexec_options
          vkexec::vkexec_warnings
          benchmark::benchmark
          benchmark::benchmark_main)
set_target_properties(vkexec_graph_planning_benchmark PROPERTIES CXX_SCAN_FOR_MODULES OFF)

add_executable(vkexec_graph_submission_benchmark graph_submission.cpp)
target_link_libraries(
  vkexec_graph_submission_benchmark
  PRIVATE vkexec::vkexec
          vkexec::features
          vkexec::vkexec_options
          vkexec::vkexec_warnings
          benchmark::benchmark
          benchmark::benchmark_main)
set_target_properties(vkexec_graph_submission_benchmark PROPERTIES CXX_SCAN_FOR_MODULES OFF)

add_executable(vkexec_graph_recording_benchmark graph_recording.cpp recording_allocations.cpp)
target_link_libraries(
  vkexec_graph_recording_benchmark
  PRIVATE vkexec::vkexec
          vkexec::vkexec_options
          vkexec::vkexec_warnings
          benchmark::benchmark
          benchmark::benchmark_main)
set_target_properties(vkexec_graph_recording_benchmark PROPERTIES CXX_SCAN_FOR_MODULES OFF)
