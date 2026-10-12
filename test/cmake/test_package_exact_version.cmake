if(NOT EXISTS "${VKEXEC_VERSION_FILE}")
  message(FATAL_ERROR "Package version file does not exist: ${VKEXEC_VERSION_FILE}")
endif()

if(NOT VKEXEC_EXPECTED_VERSION)
  message(FATAL_ERROR "VKEXEC_EXPECTED_VERSION is required")
endif()

set(VKEXEC_REJECTED_VERSIONS "0.1.0" "0.2" "0.3.0")

function(vkexec_check_package_version requested_version expected_compatible)
  set(PACKAGE_FIND_VERSION "${requested_version}")
  if(requested_version MATCHES "^([0-9]+)\\.([0-9]+)")
    set(PACKAGE_FIND_VERSION_MAJOR "${CMAKE_MATCH_1}")
    set(PACKAGE_FIND_VERSION_MINOR "${CMAKE_MATCH_2}")
  endif()
  include("${VKEXEC_VERSION_FILE}")
  if(NOT
     PACKAGE_VERSION_COMPATIBLE
     STREQUAL
     expected_compatible)
    message(
      FATAL_ERROR
        "Requested ${requested_version}: expected compatibility ${expected_compatible}, got ${PACKAGE_VERSION_COMPATIBLE}"
    )
  endif()
endfunction()

vkexec_check_package_version("${VKEXEC_EXPECTED_VERSION}" TRUE)
foreach(rejected_version IN LISTS VKEXEC_REJECTED_VERSIONS)
  vkexec_check_package_version("${rejected_version}" FALSE)
endforeach()
