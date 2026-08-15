add_executable(vkexec_sort_example main.cpp)

target_link_libraries(
  vkexec_sort_example
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec)

vkexec_configure_linker(vkexec_sort_example)
