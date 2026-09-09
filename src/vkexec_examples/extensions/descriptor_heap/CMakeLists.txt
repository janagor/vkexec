add_executable(descriptor_heap descriptor_heap.cpp)

target_link_libraries(
  descriptor_heap
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::ext_descriptor_heap)

vkexec_configure_linker(descriptor_heap)
