macro(vkexec_configure_linker project_name)
  set(vkexec_USER_LINKER_OPTION
      "DEFAULT"
      CACHE STRING "Linker to be used")
  set(vkexec_USER_LINKER_OPTION_VALUES
      "DEFAULT"
      "SYSTEM"
      "LLD"
      "GOLD"
      "BFD"
      "MOLD"
      "SOLD"
      "APPLE_CLASSIC"
      "MSVC")
  set_property(CACHE vkexec_USER_LINKER_OPTION PROPERTY STRINGS ${vkexec_USER_LINKER_OPTION_VALUES})
  list(
    FIND
    vkexec_USER_LINKER_OPTION_VALUES
    ${vkexec_USER_LINKER_OPTION}
    vkexec_USER_LINKER_OPTION_INDEX)

  if(${vkexec_USER_LINKER_OPTION_INDEX} EQUAL -1)
    message(
      STATUS
        "Using custom linker: '${vkexec_USER_LINKER_OPTION}', explicitly supported entries are ${vkexec_USER_LINKER_OPTION_VALUES}"
    )
  endif()

  set_target_properties(${project_name} PROPERTIES LINKER_TYPE "${vkexec_USER_LINKER_OPTION}")
endmacro()
