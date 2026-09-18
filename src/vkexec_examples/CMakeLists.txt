add_library(vkexec_examples_glfw INTERFACE)
target_link_libraries(vkexec_examples_glfw INTERFACE glfw)

add_subdirectory(extensions)

add_executable(compute compute.cpp)
add_executable(compute_execution compute_execution.cpp)
add_executable(tensor_sim tensor_sim.cpp)
add_executable(sort sort.cpp)
add_executable(passes passes.cpp)
add_executable(spirv spirv.cpp)
add_executable(triangle triangle.cpp)
add_executable(graphics_execution graphics_execution.cpp)
add_executable(heap_present heap_present.cpp)
add_executable(particles particles.cpp)
add_executable(layers layers.cpp)
add_executable(mesh mesh.cpp load_gltf_mesh.cpp)

target_link_libraries(compute PRIVATE vkexec::vkexec_options vkexec::vkexec_warnings vkexec::vkexec)

target_link_libraries(compute_execution PRIVATE vkexec::vkexec_options vkexec::vkexec_warnings vkexec::vkexec)

target_link_libraries(tensor_sim PRIVATE vkexec::vkexec_options vkexec::vkexec_warnings vkexec::vkexec vkexec::features)

target_link_libraries(sort PRIVATE vkexec::vkexec_options vkexec::vkexec_warnings vkexec::vkexec)

target_link_libraries(passes PRIVATE vkexec::vkexec_options vkexec::vkexec_warnings vkexec::vkexec)

target_link_libraries(spirv PRIVATE vkexec::vkexec_options vkexec::vkexec_warnings vkexec::vkexec)

target_link_libraries(
  triangle
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::vkexec_graphics
          vkexec_examples_glfw)

target_link_libraries(
  graphics_execution
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::vkexec_graphics
          vkexec_examples_glfw)

target_link_libraries(
  heap_present
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::ext_descriptor_heap
          vkexec::ext_dynamic_rendering
          vkexec::vkexec_graphics)

target_link_libraries(
  particles
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::vkexec_graphics
          vkexec_examples_glfw)

target_link_libraries(
  layers
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::vkexec_graphics
          vkexec_examples_glfw)

set(vkexec_mesh_gltf "${CMAKE_CURRENT_SOURCE_DIR}/resources/suzanne_blender_monkey/scene.gltf")

target_link_libraries(
  mesh
  PRIVATE vkexec::vkexec_options
          vkexec::vkexec_warnings
          vkexec::vkexec
          vkexec::vkexec_graphics
          vkexec_examples_glfw
          tinygltf::tinygltf)

target_compile_definitions(mesh PRIVATE "VKEXEC_MESH_GLTF_PATH=\"${vkexec_mesh_gltf}\"")

vkexec_configure_linker(compute)
vkexec_configure_linker(compute_execution)
vkexec_configure_linker(tensor_sim)
vkexec_configure_linker(sort)
vkexec_configure_linker(passes)
vkexec_configure_linker(spirv)
vkexec_configure_linker(triangle)
vkexec_configure_linker(graphics_execution)
vkexec_configure_linker(heap_present)
vkexec_configure_linker(particles)
vkexec_configure_linker(layers)
vkexec_configure_linker(mesh)
