if(NOT DEFINED VKEXEC_SOURCE_DIR)
  message(FATAL_ERROR "VKEXEC_SOURCE_DIR is required")
endif()

file(
  GLOB_RECURSE _docs
  LIST_DIRECTORIES FALSE
  "${VKEXEC_SOURCE_DIR}/docs/*.md")
list(APPEND _docs "${VKEXEC_SOURCE_DIR}/README.md")
foreach(_doc IN LISTS _docs)
  file(READ "${_doc}" _content)
  foreach(
    _stale IN
    ITEMS "cmake_template"
          "/workspaces/cmake_template"
          "C++17"
          "gcc 7+"
          "clang 6+"
          "Visual Studio 2019"
          ".allocator = vma")
    string(FIND "${_content}" "${_stale}" _position)
    if(NOT
       _position
       EQUAL
       -1)
      message(FATAL_ERROR "Stale documentation string '${_stale}' in ${_doc}")
    endif()
  endforeach()
endforeach()

file(GLOB _old_readmes "${VKEXEC_SOURCE_DIR}/README_*.md")
if(_old_readmes)
  message(FATAL_ERROR "Obsolete root READMEs remain: ${_old_readmes}")
endif()
