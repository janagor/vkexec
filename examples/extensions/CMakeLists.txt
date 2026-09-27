add_subdirectory(descriptor_heap)
add_subdirectory(dynamic_rendering)

target_link_libraries(descriptor_heap PRIVATE vkexec::vma)
target_link_libraries(dynamic_rendering PRIVATE vkexec::vma)
