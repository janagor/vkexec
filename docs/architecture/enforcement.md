# Architecture enforcement

CTest registers `CheckStdexecPrivateApi.cmake`, `CheckVmaBoundary.cmake`, and `CheckVkBootstrapBoundary.cmake`. These checks guard the stdexec-private, allocator, and device-creation dependency boundaries. They complement ordinary behavioral tests; a passing build alone does not establish those architecture rules.
