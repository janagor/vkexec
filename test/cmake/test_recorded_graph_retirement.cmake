execute_process(
  COMMAND "${VKEXEC_RETIREMENT_HELPER}"
  RESULT_VARIABLE outcome
  OUTPUT_VARIABLE output
  ERROR_VARIABLE errors)
if(outcome STREQUAL "125")
  message("SKIP: Vulkan is unavailable")
elseif(
  NOT
  outcome
  STREQUAL
  "77")
  message(FATAL_ERROR "Unretired recording did not trigger the retirement guard (${outcome}): ${output}${errors}")
endif()
