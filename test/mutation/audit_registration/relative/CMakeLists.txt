# The relative YAML is resolved here, while finalization runs in the parent.
vkexec_register_mutation_scope(
  NAME
  other
  CONFIG
  "../inactive.yml"
  TEST_TARGET
  dummy
  SOURCES
  dummy.cpp
  TEST_ARGS
  "inactive-argument")
