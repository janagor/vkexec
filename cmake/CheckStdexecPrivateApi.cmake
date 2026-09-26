if(NOT DEFINED VKEXEC_SOURCE_DIR)
  message(FATAL_ERROR "VKEXEC_SOURCE_DIR is required")
endif()

file(
  GLOB_RECURSE vkexec_source_files
  LIST_DIRECTORIES FALSE
  "${VKEXEC_SOURCE_DIR}/include/*.h"
  "${VKEXEC_SOURCE_DIR}/include/*.hpp"
  "${VKEXEC_SOURCE_DIR}/include/*.hh"
  "${VKEXEC_SOURCE_DIR}/include/*.hxx"
  "${VKEXEC_SOURCE_DIR}/include/*.inl"
  "${VKEXEC_SOURCE_DIR}/include/*.ipp"
  "${VKEXEC_SOURCE_DIR}/include/*.tpp"
  "${VKEXEC_SOURCE_DIR}/src/*.c"
  "${VKEXEC_SOURCE_DIR}/src/*.cc"
  "${VKEXEC_SOURCE_DIR}/src/*.cpp"
  "${VKEXEC_SOURCE_DIR}/src/*.cxx"
  "${VKEXEC_SOURCE_DIR}/src/*.inl"
  "${VKEXEC_SOURCE_DIR}/src/*.ipp"
  "${VKEXEC_SOURCE_DIR}/src/*.tpp"
  "${VKEXEC_SOURCE_DIR}/test/*.c"
  "${VKEXEC_SOURCE_DIR}/test/*.cc"
  "${VKEXEC_SOURCE_DIR}/test/*.cpp"
  "${VKEXEC_SOURCE_DIR}/test/*.cxx"
  "${VKEXEC_SOURCE_DIR}/test/*.inl"
  "${VKEXEC_SOURCE_DIR}/test/*.ipp"
  "${VKEXEC_SOURCE_DIR}/test/*.tpp"
  "${VKEXEC_SOURCE_DIR}/examples/*.c"
  "${VKEXEC_SOURCE_DIR}/examples/*.cc"
  "${VKEXEC_SOURCE_DIR}/examples/*.cpp"
  "${VKEXEC_SOURCE_DIR}/examples/*.cxx"
  "${VKEXEC_SOURCE_DIR}/examples/*.inl"
  "${VKEXEC_SOURCE_DIR}/examples/*.ipp"
  "${VKEXEC_SOURCE_DIR}/examples/*.tpp"
  "${VKEXEC_SOURCE_DIR}/bench/*.h"
  "${VKEXEC_SOURCE_DIR}/bench/*.hpp"
  "${VKEXEC_SOURCE_DIR}/bench/*.hh"
  "${VKEXEC_SOURCE_DIR}/bench/*.hxx"
  "${VKEXEC_SOURCE_DIR}/bench/*.inl"
  "${VKEXEC_SOURCE_DIR}/bench/*.ipp"
  "${VKEXEC_SOURCE_DIR}/bench/*.tpp"
  "${VKEXEC_SOURCE_DIR}/bench/*.c"
  "${VKEXEC_SOURCE_DIR}/bench/*.cc"
  "${VKEXEC_SOURCE_DIR}/bench/*.cpp"
  "${VKEXEC_SOURCE_DIR}/bench/*.cxx"
  "${VKEXEC_SOURCE_DIR}/benchmark/*.h"
  "${VKEXEC_SOURCE_DIR}/benchmark/*.hpp"
  "${VKEXEC_SOURCE_DIR}/benchmark/*.hh"
  "${VKEXEC_SOURCE_DIR}/benchmark/*.hxx"
  "${VKEXEC_SOURCE_DIR}/benchmark/*.inl"
  "${VKEXEC_SOURCE_DIR}/benchmark/*.ipp"
  "${VKEXEC_SOURCE_DIR}/benchmark/*.tpp"
  "${VKEXEC_SOURCE_DIR}/benchmark/*.c"
  "${VKEXEC_SOURCE_DIR}/benchmark/*.cc"
  "${VKEXEC_SOURCE_DIR}/benchmark/*.cpp"
  "${VKEXEC_SOURCE_DIR}/benchmark/*.cxx")

set(stdexec_compat_header "${VKEXEC_SOURCE_DIR}/include/vkexec/detail/stdexec_compat.hpp")
set(private_stdexec_pattern "(stdexec|ex|exec|STDEXEC)::__|stdexec/__detail/")

foreach(vkexec_source_file IN LISTS vkexec_source_files)
  if(vkexec_source_file STREQUAL stdexec_compat_header)
    continue()
  endif()

  file(READ "${vkexec_source_file}" vkexec_source_contents)
  if(vkexec_source_contents MATCHES "${private_stdexec_pattern}")
    message(
      FATAL_ERROR
        "vkexec must not depend directly on NVIDIA/stdexec private APIs.\n"
        "Move unavoidable compatibility code to:\n"
        "  include/vkexec/detail/stdexec_compat.hpp\n"
        "Offending file: ${vkexec_source_file}")
  endif()
endforeach()
