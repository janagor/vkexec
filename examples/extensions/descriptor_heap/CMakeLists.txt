add_executable(descriptor_heap descriptor_heap.cpp)
set_target_properties(descriptor_heap PROPERTIES CXX_SCAN_FOR_MODULES OFF)

target_link_libraries(
  descriptor_heap
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::ext_descriptor_heap
          vkexec::tools)

vkexec_configure_linker(descriptor_heap)
